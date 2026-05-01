import os
import re
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset

class RFIDDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, name="rfid"):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        data_list = []
        raw_dir = os.path.join(self.root, 'raw')
        
        # Regex to capture participant ID from filenames like pP01_...
        p_regex = re.compile(r'pP(\d+)')
        
        # We need to sort to ensure consistent indexing
        all_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.pt')])
        
        print(f"Found {len(all_files)} raw files. Processing...")
        
        for filename in all_files:
            # Skip files that don't have a clear participant ID (like unlabelled ones)
            match = p_regex.search(filename)
            if not match:
                continue
                
            p_id = int(match.group(1)) - 1 # 0-indexed
            
            path = os.path.join(raw_dir, filename)
            # Load the raw 8x60 tensor
            try:
                arr = torch.load(path, weights_only=False)
                if isinstance(arr, np.ndarray):
                    arr = torch.from_numpy(arr)
                
                # Basic Normalization
                arr_mean = arr.mean()
                arr_std = arr.std() + 1e-7
                arr = (arr - arr_mean) / arr_std
                
                # Standard 8-tag sparse edges (0-1, 2-3, etc)
                # Note: Our training script will overwrite this with Fully Connected edges later
                edge_index = torch.tensor([[0, 1, 1, 0, 2, 3, 3, 2, 4, 5, 5, 4, 6, 7, 7, 6],
                                          [1, 0, 0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7]], dtype=torch.long)
                
                # Gesture label (y) and Participant label (p_y)
                # Assuming Jeremias format: pP[Participant]_G[Gesture]_...
                g_match = re.search(r'_G(\d+)', filename)
                if not g_match: continue
                g_id = int(g_match.group(1)) - 1
                
                data = Data(x=arr.float(), 
                            edge_index=edge_index, 
                            y=torch.tensor([g_id], dtype=torch.long),
                            p_y=torch.tensor([p_id], dtype=torch.long))
                
                data_list.append(data)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def main():
    root_dir = "/root/Adversarial_Learning/Jeremias/codebase/AdversarialLearningProject/SavedTensor"
    
    if not os.path.exists(root_dir):
        print(f"ERROR: {root_dir} not found.")
        return

    # This will now run without importing ANY GNNPlus modules
    dataset = RFIDDataset(root=root_dir)
    
    print(f"\nSUCCESS! Created: {dataset.processed_paths[0]}")
    print(f"Total samples: {len(dataset)}")

if __name__ == "__main__":
    main()
