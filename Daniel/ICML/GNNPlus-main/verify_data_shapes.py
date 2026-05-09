import torch
from torch_geometric.data import Data

data_path = r'C:\Users\PC\Desktop\AL\ICML\GNNPlus-main\RFIDDataSet\processed\geometric_data_windowed_2.pt'
data, slices = torch.load(data_path, weights_only=False)

# Check first sample
x = data.x[slices['x'][0]:slices['x'][1]]
edge_index = data.edge_index[:, slices['edge_index'][0]:slices['edge_index'][1]]
window_id = data.window_id[slices['window_id'][0]:slices['window_id'][1]]

print(f"Num nodes: {x.shape[0]}")
print(f"Num features: {x.shape[1]}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Window ID shape: {window_id.shape}")
print(f"Unique window IDs: {torch.unique(window_id)}")
