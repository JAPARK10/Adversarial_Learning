import torch
import torch.nn.functional as F
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from GNNPlus.layer.gatedgcn_layer import GatedGCNLayer
from GNNPlus.layer.gine_conv_layer import GINEConvLayer
from GNNPlus.layer.grl import GradientReversalLayer
import os
from dotenv import load_dotenv

# Load toggle from .env
load_dotenv()
USE_ADVERSARIAL = os.getenv("USE_ADVERSARIAL_LAYERS", "false").lower() == "true"
USE_CONTRASTIVE = os.getenv("USE_CONTRASTIVE_LEARNING", "false").lower() == "true"

@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        
        # # Due to GatedGcn
        # if cfg.dataset.node_encoder:
        #     self.encoder = FeatureEncoder(dim_in)
        #     dim_in = self.encoder.dim_in
        # else:
        #     self.encoder = torch.nn.Identity()
        
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual,ffn=cfg.gnn.ffn))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        # Adversarial Branch
        if USE_ADVERSARIAL:
            print("[*] Building Adversarial Branch (Participant Discriminator)")
            self.grl = GradientReversalLayer(alpha=1.0)
            # Assuming 17 participants based on the PDF/DataSet3m
            self.participant_discriminator = torch.nn.Sequential(
                torch.nn.Linear(cfg.gnn.dim_inner, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 18) # Changed to 18 to be safe with 1-based indexing or extras
            )

        # Contrastive Branch (Projection Head)
        if USE_CONTRASTIVE:
            print("[*] Building Contrastive Branch (Projection Head)")
            self.projection_head = torch.nn.Sequential(
                torch.nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                torch.nn.ReLU(),
                torch.nn.Linear(cfg.gnn.dim_inner, 128) # 128-dim embedding for SupCon
            )

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcn':
            return GatedGCNLayer
        elif model_type == 'gine':
            return GINEConvLayer
        elif model_type == 'gcn':
            from GNNPlus.layer.gcn_conv_layer import GCNConvLayer
            return GCNConvLayer
        elif model_type == 'gcne':
            from GNNPlus.layer.gcn_conv_layer_e import GCNConvLayer
            return GCNConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        # 1. Encoder
        batch = self.encoder(batch)
        
        # 2. Pre-MP (Optional Linear)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
            
        # 3. GNN Layers (Core Convolution)
        batch = self.gnn_layers(batch)
        
        # 4. Exercise Head (Main Prediction)
        # We need the pooled graph features for the heads
        # Usually head_dict[cfg.gnn.head] handles pooling. 
        # Let's check how GraphGym heads work.
        # They usually return (pred, true)
        
        # We need the node features pooled to graph features 
        # (Only calculate once if either branch is active)
        if USE_ADVERSARIAL or USE_CONTRASTIVE:
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(batch.x, batch.batch)
        
        # 4. Exercise Head (Main Prediction)
        exercise_pred, true = self.post_mp(batch)
        
        preds = {'exercise': exercise_pred}
        
        if USE_ADVERSARIAL:
            # Pass through GRL (flips gradient during backprop)
            reverse_emb = self.grl(graph_emb)
            preds['participant'] = self.participant_discriminator(reverse_emb)
        
        if USE_CONTRASTIVE:
            # Normalize projection embedding for contrastive distance
            con_emb = self.projection_head(graph_emb)
            preds['contrastive'] = F.normalize(con_emb, dim=1)
            
        return preds, true
