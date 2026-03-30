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
        print("RAW DIR EXISTS:", os.path.exists(self.raw_dir))
        print("RAW DIR CONTENT:", os.listdir(self.raw_dir))

        data_list = []

        class_folders = sorted(os.listdir(self.raw_dir))
        class_to_label = {cls: i for i, cls in enumerate(class_folders)}

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

                arr = np.load(osp.join(cls_path, file))  # (30, 8, 2)

                # Build node features
                node_features = []
                for tag in range(8):
                    tag_signal = arr[:, tag, :]      # (30, 2)
                    tag_feat = tag_signal.reshape(-1)  # (60,)
                    node_features.append(tag_feat)

                x = torch.tensor(node_features, dtype=torch.float)
                y = torch.tensor([label], dtype=torch.long)


                # # (REQUIRED for GatedGCN) --Should be comment for Gcn
                # num_edges = edge_index.size(1)
                # # edge_attr = torch.ones(num_edges, x.size(1))
                # edge_attr = torch.ones(edge_index.size(1), 1)  

                # For Gcn

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y
                )

                # For GatedGcn
                # data = Data(
                #     x=x,
                #     edge_index=edge_index,
                #     edge_attr=edge_attr,
                #     y=y
                # )

                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
