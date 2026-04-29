import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class RFIDDataset(InMemoryDataset):
    def __init__(self, root=r'/home/golipos1/GNNPlus/GNNPlus-main/RFIDDataSet/DataSet1', name=None,
                 transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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
        print("RAW DIR EXISTS:", os.path.exists(self.raw_dir))
        print("RAW DIR CONTENT:", os.listdir(self.raw_dir))

        data_list = []

        class_folders = sorted(os.listdir(self.raw_dir))
        class_to_label = {cls: i for i, cls in enumerate(class_folders)}
        user_to_idx = {}

        # Fixed physical proximity edges
        edge_pairs = [
            (0, 1), (1, 0),
            (2, 3), (3, 2),
            (4, 5), (5, 4),
            (6, 7), (7, 6),
        ]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t()

        for cls in class_folders:
            cls_path = osp.join(self.raw_dir, cls)
            if not osp.isdir(cls_path):
                continue

            label = class_to_label[cls]
            for file in tqdm(os.listdir(cls_path), desc=f'Processing {cls}'):
                if not file.endswith('.npy'):
                    continue

                # subj_{subject}_gest_{target_gesture_dir_name}_idx_{sample_idx:05d}.npy
                # Extract subject from filename
                parts = file.split('_')
                if len(parts) >= 2 and parts[0] == 'subj':
                    subject = parts[1]
                else:
                    subject = 'unknown'
                    
                if subject not in user_to_idx:
                    user_to_idx[subject] = len(user_to_idx)

                arr = np.load(osp.join(cls_path, file))  # (30, 8, 2)

                # 1. Node features: 240 nodes, each with 2 features (RSS, phase)
                x = torch.tensor(arr.reshape(-1, 2), dtype=torch.float)

                edges_source = []
                edges_target = []

                # 2. Add spatial edges within each timestamp
                spatial_pairs = [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]
                for t in range(30):
                    for src, dst in spatial_pairs:
                        edges_source.append(t * 8 + src)
                        edges_target.append(t * 8 + dst)

                # 3. Add temporal K-NN edges across consecutive timestamps
                K = 3
                for t in range(1, 30):
                    prev_feats = x[(t - 1) * 8 : t * 8]  # Shape: (8, 2)
                    curr_feats = x[t * 8 : (t + 1) * 8]  # Shape: (8, 2)

                    distances = torch.cdist(curr_feats, prev_feats, p=2.0)
                    for i in range(8):
                        _, topk_indices = torch.topk(distances[i], K, largest=False)
                        for neighbor_idx in topk_indices:
                            # Directed edge from t-1 (neighbor) to t (i)
                            u = (t - 1) * 8 + neighbor_idx.item()
                            v = t * 8 + i
                            edges_source.append(u)
                            edges_target.append(v)

                edge_index = torch.tensor([edges_source, edges_target], dtype=torch.long)

                y = torch.tensor([label], dtype=torch.long)
                y_user = torch.tensor([user_to_idx[subject]], dtype=torch.long)

                # For Gcn
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    y_user=y_user,
                    subject=subject
                )

                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
