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
        
        # Regex to capture participant ID (e.g., _p00) and Gesture ID (from folder name)
        p_regex = re.compile(r'_p(\d+)')
        g_regex = re.compile(r'gesture(\d+)')
        
        print(f"Scanning subdirectories in: {self.root}...")
        
        # Walk through all subdirectories
        all_files = []
        for root, dirs, files in os.walk(self.root):
            if 'processed' in root: continue
            for f in files:
                # We only want the files that include the participant tag (_pXX)
                if f.endswith('.npy') and '_p' in f:
                    all_files.append(os.path.join(root, f))
        
        all_files.sort()
        print(f"Found {len(all_files)} raw .npy samples. Generating Super-Dataset (3x Augmentation)...")
        
        num_nodes = 8
        adj = torch.ones((num_nodes, num_nodes))
        fc_edge_index = adj.nonzero().t().contiguous()

        for i, path in enumerate(tqdm(all_files)):
            filename = os.path.basename(path)
            dirname = os.path.basename(os.path.dirname(path))
            
            p_match = p_regex.search(filename)
            g_match = g_regex.search(dirname)
            
            if not p_match or not g_match:
                if i < 5: print(f"Regex fail: {filename} in {dirname}")
                continue
                
            p_id = int(p_match.group(1)) # Assuming 0-indexed already
            g_id = int(g_match.group(1)) - 1 # Assuming folder starts at gesture1
            
            try:
                raw_np = np.load(path)
                # Handle the actual shape: (30 timesteps, 8 tags, 2 features)
                if raw_np.shape != (30, 8, 2):
                    if i < 5: print(f"Unexpected shape: {filename} is {raw_np.shape}")
                    continue
                
                # Convert to Torch and Reshape: (30, 8, 2) -> (8, 60)
                # We want each of the 8 tags to have 60 features (30 RSSI + 30 Phase)
                raw_tensor = torch.from_numpy(raw_np).float()
                rssi = raw_tensor[:, :, 0].permute(1, 0)   # (8, 30)
                phase = raw_tensor[:, :, 1].permute(1, 0)  # (8, 30)
                reshaped_x = torch.cat([rssi, phase], dim=1) # (8, 60)

                # --- 1. Original Sample ---
                data_list.append(self.create_data_object(reshaped_x, fc_edge_index, g_id, p_id))

                # --- 2. Augmented: Gaussian Noise (5%) ---
                noise = torch.randn_like(reshaped_x) * 0.05
                data_list.append(self.create_data_object(reshaped_x + noise, fc_edge_index, g_id, p_id))

                # --- 3. Augmented: Random Scaling (0.9 to 1.1) ---
                scale = 0.9 + (torch.rand(1) * 0.2)
                data_list.append(self.create_data_object(reshaped_x * scale, fc_edge_index, g_id, p_id))

            except Exception as e:
                if i < 5: print(f"Error processing {filename}: {e}")

        print(f"Collatting {len(data_list)} samples...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def create_data_object(self, x, edge_index, g_id, p_id):
        # x is [8, 60] -> 30 RSSI, 30 Phase
        rssi = x[:, :30]
        phase = x[:, 30:]
        
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
    root_dir = "/root/Adversarial_Learning/Jeremias/codebase/AdversarialLearningProject/SavedTensor"
    
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found.")
        return

    dataset = SuperGeometricDataset(root=root_dir)
    print(f"\nDONE! Super-Dataset created at: {dataset.processed_paths[0]}")
    print(f"Total samples: {len(dataset)}")

if __name__ == "__main__":
    main()
