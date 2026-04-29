import os
import os.path as osp
import re
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def _participant_from_filename(fname: str) -> int:
    fname = os.path.basename(fname)
    m = re.match(r'^p(\d+)_', fname, re.IGNORECASE)
    if m:
        return int(m.group(1)) - 1
    return 0


class RFIDDataset(InMemoryDataset):
    def __init__(self, root=r'/home/golipos1/GNNPlus/GNNPlus-main/RFIDDataSet/DataSet1',
                 name=None, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def download(self):
        pass

    def process(self):
        print("RAW DIR:", self.raw_dir)
        data_list = []
        class_folders = sorted(os.listdir(self.raw_dir))
        class_to_label = {cls: i for i, cls in enumerate(class_folders)}
        edge_pairs = [
            (0,1),(1,0),(2,3),(3,2),
            (4,5),(5,4),(6,7),(7,6),
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
                arr = np.load(osp.join(cls_path, file))
                node_features = []
                for tag in range(8):
                    tag_feat = arr[:, tag, :].reshape(-1)
                    node_features.append(tag_feat)
                x = torch.tensor(node_features, dtype=torch.float)
                y = torch.tensor([label], dtype=torch.long)
                pid = _participant_from_filename(file)
                participant = torch.tensor([pid], dtype=torch.long)
                data = Data(x=x, edge_index=edge_index,
                            y=y, participant=participant)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])