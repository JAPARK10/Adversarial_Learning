import torch.nn as nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import scatter
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import torch
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)


class GCNConvWithEdges(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, bias=True):
        super(GCNConvWithEdges, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
        else:
            self.lin_edge = None
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        # 1. Linear transformation
        x = self.lin(x)

        # 2. GCN Normalization
        if isinstance(edge_index, Tensor):
            num_nodes = x.size(0)
            edge_index, _ = pyg_nn.conv.gcn_conv.gcn_norm(
                edge_index, None, num_nodes, False, False, self.flow, x.dtype)

        # 3. Propagate
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # 4. Bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is not None and self.lin_edge is not None:
            # Project edge features to match node feature dimension
            return (x_j + self.lin_edge(edge_attr)).relu()
        return x_j.relu()
    
class GCNConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual, ffn):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = True
        self.ffn = ffn
        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(dim_out)
        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        # Get edge dimension from config
        edge_dim = cfg.dataset.get('edge_dim', None)
        print(f"[MODEL CHECK] Instantiating GCNConvWithEdges: node_dim={dim_in}->{dim_out}, edge_dim={edge_dim}")
        
        self.model = GCNConvWithEdges(dim_in, dim_out, edge_dim, bias=True)
        
        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out*2)
            self.ff_linear2 = nn.Linear(dim_out*2, dim_out)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)
        
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x_in = batch.x
        
        # Pass edge_attr if present
        edge_attr = getattr(batch, 'edge_attr', None)
        
        batch.x = self.model(batch.x, batch.edge_index, edge_attr=edge_attr)
        if self.batch_norm:
            batch.x = self.bn_node_x(batch.x)
        batch.x = self.act(batch.x)
        
        if self.residual:
            batch.x = x_in + batch.x  # Residual connection.
        
        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)
            
            batch.x = batch.x + self._ff_block(batch.x)

            if self.batch_norm:
                batch.x = self.norm2(batch.x)

        return batch
    
