import os
import re
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

class SuperGeometricDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['super_geometric_data.pt']

    def process(self):
        data_list = []
        
        # Regex to capture participant ID (e.g., p01) and Gesture ID (from folder name)
        p_regex = re.compile(r'p(\d+)')
        g_regex = re.compile(r'gesture(\d+)')
        
        print(f"Scanning subdirectories in: {self.root}...")
        
        # Walk through all subdirectories
        all_files = []
        for root, dirs, files in os.walk(self.root):
            if 'processed' in root: continue
            for f in files:
                if f.endswith('.npy'):
                    all_files.append(os.path.join(root, f))
        
        all_files.sort()
        print(f"Found {len(all_files)} raw .npy samples. Generating Super-Dataset (10x Augmentation + Phase Unwrapping)...")
        
        num_nodes = 8
        adj = torch.ones((num_nodes, num_nodes))
        fc_edge_index = adj.nonzero().t().contiguous()

        for i, path in enumerate(tqdm(all_files)):
            filename = os.path.basename(path)
            dirname = os.path.basename(os.path.dirname(path))
            
            p_match = p_regex.search(filename)
            g_match = g_regex.search(dirname)
            
            if not p_match or not g_match:
                continue
                
            p_id = int(p_match.group(1)) - 1
            g_id = int(g_match.group(1)) - 1 
            
            try:
                raw_np = np.load(path)
                if raw_np.shape != (30, 8, 2):
                    continue
                
                raw_tensor = torch.from_numpy(raw_np).float()
                rssi = raw_tensor[:, :, 0].permute(1, 0)   # (8, 30)
                
                # --- PHASE UNWRAPPING (The Fix) ---
                phase_raw = raw_tensor[:, :, 1].permute(1, 0).numpy()
                phase_unwrapped = np.unwrap(phase_raw, axis=1) 
                phase = torch.from_numpy(phase_unwrapped).float()
                
                reshaped_x = torch.cat([rssi, phase], dim=1) # (8, 60)

                # --- 10x AUGMENTATION ---
                # 1. Original
                data_list.append(self.create_data_object(reshaped_x, fc_edge_index, g_id, p_id))

                # 2-4. Multi-Scale Noise
                for sigma in [0.01, 0.05, 0.1]:
                    noise = torch.randn_like(reshaped_x) * sigma
                    data_list.append(self.create_data_object(reshaped_x + noise, fc_edge_index, g_id, p_id))

                # 5-6. Scaling
                for scale in [0.9, 1.1]:
                    data_list.append(self.create_data_object(reshaped_x * scale, fc_edge_index, g_id, p_id))

                # 7-8. Time-Shifting
                for shift in [1, -1]:
                    rssi_s = torch.roll(rssi, shifts=shift, dims=1)
                    phase_s = torch.roll(phase, shifts=shift, dims=1)
                    data_list.append(self.create_data_object(torch.cat([rssi_s, phase_s], dim=1), fc_edge_index, g_id, p_id))

                # 9. Mirroring (Swap Arms)
                mirror_idx = [4, 5, 6, 7, 0, 1, 2, 3]
                data_list.append(self.create_data_object(reshaped_x[mirror_idx], fc_edge_index, g_id, p_id))

                # 10. Heavy Hybrid
                hybrid = (reshaped_x + torch.randn_like(reshaped_x)*0.03) * 1.05
                data_list.append(self.create_data_object(hybrid, fc_edge_index, g_id, p_id))

            except Exception:
                pass

        print(f"Collatting {len(data_list)} samples...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def create_data_object(self, x, edge_index, g_id, p_id):
        rssi = x[:, :30]
        phase = x[:, 30:]
        
        # Now deltas are smooth thanks to np.unwrap
        rssi_delta = torch.cat([torch.zeros((8, 1)), rssi[:, 1:] - rssi[:, :-1]], dim=1)
        phase_delta = torch.cat([torch.zeros((8, 1)), phase[:, 1:] - phase[:, :-1]], dim=1)
        
        combined_features = torch.cat([rssi, phase, rssi_delta, phase_delta], dim=1)
        
        mean = combined_features.mean()
        std = combined_features.std() + 1e-7
        combined_features = (combined_features - mean) / std
        
        return Data(x=combined_features, 
                    edge_index=edge_index, 
                    y=torch.tensor([g_id], dtype=torch.long),
                    p_y=torch.tensor([p_id], dtype=torch.long))

def main():
    root_dir = r"c:\Users\jerem\Desktop\Workspace_VSCode\CoDaS\Adversarial_Learning\Farhan\ICML\GNNPlus-main\RFIDDataSet"
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found.")
        return
    dataset = SuperGeometricDataset(root=root_dir)
    print(f"\nDONE! Super-Dataset created at: {dataset.processed_paths[0]}")
    print(f"Total samples: {len(dataset)}")

if __name__ == "__main__":
    main()
