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
        p_regex = re.compile(r'p(\d+)'); g_regex = re.compile(r'gesture(\d+)')
        all_files = [os.path.join(r, f) for r, d, fs in os.walk(self.root) for f in fs if f.endswith('.npy') and 'processed' not in r]
        all_files.sort()
        
        print(f"Generating 10x Raw-Chaos Dataset (The Subject 10 Formula)...")
        num_nodes = 8; fc_edge_index = torch.ones((8, 8)).nonzero().t().contiguous()

        for i, path in enumerate(tqdm(all_files)):
            filename = os.path.basename(path); dirname = os.path.basename(os.path.dirname(path))
            p_match = p_regex.search(filename); g_match = g_regex.search(dirname)
            if not p_match or not g_match: continue
            p_id = int(p_match.group(1)) - 1; g_id = int(g_match.group(1)) - 1 
            
            try:
                raw_np = np.load(path)
                if raw_np.shape != (30, 8, 2): continue
                raw_tensor = torch.from_numpy(raw_np).float()
                rssi = raw_tensor[:, :, 0].permute(1, 0)
                phase = raw_tensor[:, :, 1].permute(1, 0) # RAW PHASE

                def create_x(r, p): return torch.cat([r, p], dim=1)
                def add_obj(r, p, is_o=False): data_list.append(self.create_data_object(create_x(r, p), fc_edge_index, g_id, p_id, is_orig=is_o))

                # 3x ULTRA-FAST AUGMENTATION (Original + Mirror + 0.02 Jitter)
                add_obj(rssi, phase, is_o=True) # Original
                
                # Mirroring
                m = [4, 5, 6, 7, 0, 1, 2, 3]
                add_obj(rssi[m], phase[m])
                
                # Jitter (0.02 only)
                add_obj(rssi + torch.randn_like(rssi)*0.02, phase)

            except Exception: pass

        print(f"Collatting {len(data_list)} samples...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def create_data_object(self, x, edge_index, g_id, p_id, is_orig=False):
        rssi = x[:, :30]; phase = x[:, 30:]
        rssi_delta = torch.cat([torch.zeros((8, 1)), rssi[:, 1:] - rssi[:, :-1]], dim=1)
        phase_delta = torch.cat([torch.zeros((8, 1)), phase[:, 1:] - phase[:, :-1]], dim=1)
        combined_features = torch.cat([rssi, phase, rssi_delta, phase_delta], dim=1)
        mean = combined_features.mean(); std = combined_features.std() + 1e-7
        combined_features = (combined_features - mean) / std
        return Data(x=combined_features, edge_index=edge_index, y=torch.tensor([g_id], dtype=torch.long),
                    p_y=torch.tensor([p_id], dtype=torch.long), is_orig=torch.tensor([1 if is_orig else 0], dtype=torch.bool))

def main():
    root_dir = r"c:\Users\jerem\Desktop\Workspace_VSCode\CoDaS\Adversarial_Learning\Farhan\ICML\GNNPlus-main\RFIDDataSet"
    if not os.path.exists(root_dir): return
    dataset = SuperGeometricDataset(root=root_dir)
    print(f"\nDONE! Total samples: {len(dataset)}")

if __name__ == "__main__":
    main()
