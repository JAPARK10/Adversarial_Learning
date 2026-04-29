import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from GNNPlus.layer.gatedgcn_layer import GatedGCNLayer
from GNNPlus.layer.gine_conv_layer import GINEConvLayer

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

        # Adversarial Head
        if hasattr(cfg, 'adv') and cfg.adv.use:
            from GNNPlus.layer.grl import GRL
            self.grl = GRL(lambda_u=cfg.adv.lambda_u)
            self.adv_head = torch.nn.Linear(in_features=cfg.gnn.dim_inner, out_features=cfg.adv.num_users)
        else:
            self.adv_head = None

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
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        batch = self.gnn_layers(batch)

        y_user = batch.y_user if hasattr(batch, 'y_user') else None

        # 1. Pool node embeddings to get graph embedding f_E
        f_E = self.post_mp.pooling_fun(batch.x, batch.batch)

        # Helper to extract MLP from different GraphGym heads
        def get_mlp(head):
            if isinstance(head, torch.nn.Linear):
                return head
            if hasattr(head, 'mlp'):
                return head.mlp
            elif hasattr(head, 'layer_post_mp'):
                return head.layer_post_mp
            else:
                raise AttributeError("Head missing mlp components")

        # 2. Gesture branch: f_E -> gesture MLP
        pred = get_mlp(self.post_mp)(f_E)
        if hasattr(self.post_mp, '_scale_and_shift'):
            pred = self.post_mp._scale_and_shift(pred)

        # 3. Adversarial branch: f_E -> GRL -> user MLP
        if self.adv_head is not None:
            f_E_adv = self.grl(f_E)
            pred_adv = get_mlp(self.adv_head)(f_E_adv)
            if hasattr(self.adv_head, '_scale_and_shift'):
                pred_adv = self.adv_head._scale_and_shift(pred_adv)
            return pred, batch.y, pred_adv, y_user
        else:
            return pred, batch.y
