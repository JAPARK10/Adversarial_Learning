import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class RFIDDataset(InMemoryDataset):
    def __init__(self, root=None, name=None,
                 transform=None, pre_transform=None):
        self.name = name
        if root is None:
            root = r'codebase/AdversarialLearningProject (1)/ICML/GNNPlus-main/RFIDDataSet'
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # raw data are class folders, so we just check existence
        return []

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def download(self):
        # data already exists locally
        pass

    def process(self):
        print("RAW DIR:", self.raw_dir)
        # In the new SavedTensor structure, gesture folders are directly in RAW DIR
        # gesture1, gesture2, ...
        gesture_folders = sorted([d for d in os.listdir(self.raw_dir) 
                                if os.path.isdir(osp.join(self.raw_dir, d))])
        
        print(f"[*] Found {len(gesture_folders)} gesture folders.")
        data_list = []

        # Fixed physical proximity edges
        edge_pairs = [
            (0, 1), (1, 0),
            (2, 3), (3, 2),
            (4, 5), (5, 4),
            (6, 7), (7, 6),
        ]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t()

        for g_folder in gesture_folders:
            # Assuming folder name is 'gesture1', 'gesture2' or just the label index
            try:
                # Extract digits if it's 'gesture1' -> 1
                import re
                match = re.search(r'\d+', g_folder)
                label = int(match.group()) if match else int(g_folder)
            except:
                label = 0 # fallback
                
            g_path = osp.join(self.raw_dir, g_folder)
            
            for file in tqdm(os.listdir(g_path), desc=f'Loading {g_folder}'):
                if not file.endswith('.npy'):
                    continue

                arr = np.load(osp.join(g_path, file))  # (30, 8, 2)
                
                # Build node features (30 timesteps * 2 features = 60 per tag)
                node_features = []
                for tag in range(8):
                    tag_signal = arr[:, tag, :]
                    tag_feat = tag_signal.reshape(-1)
                    node_features.append(tag_feat)

                x = torch.tensor(node_features, dtype=torch.float)
                y = torch.tensor([label], dtype=torch.long)
                
                # Extract participant ID from filename if it exists (e.g. _p01.npy)
                # Otherwise default to 0
                p_id = 0
                if "_p" in file:
                    try:
                        p_id = int(file.split("_p")[1].split(".")[0])
                    except:
                        pass
                
                p_y = torch.tensor([p_id], dtype=torch.long)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    p_y=p_y
                )
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
