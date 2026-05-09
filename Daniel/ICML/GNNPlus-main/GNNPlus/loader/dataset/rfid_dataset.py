import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg


class RFIDDataset(InMemoryDataset):
    def __init__(self, root=r'/home/golipos1/GNNPlus/GNNPlus-main/RFIDDataSet/DataSet1', name=None,
                 transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def num_edge_features(self) -> int:
        return 8

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def download(self):
        pass

    def process(self):
        print(f"[*] Processing RFID data. Saving to: {self.processed_paths[0]}")

        data_list = []
        class_folders = sorted(os.listdir(self.raw_dir))
        class_to_label = {cls: i for i, cls in enumerate(class_folders)}

        # Standard Bidirectional Paired Edges
        edge_pairs = [
            (0, 1), (1, 0),
            (2, 3), (3, 2),
            (4, 5), (5, 4),
            (6, 7), (7, 6),
        ]
        edge_index_single = torch.tensor(edge_pairs, dtype=torch.long).t()  # (2, 16)

        # 1. Edge Feature: Pair Identity (4-D One-Hot)
        edge_pairs_list = edge_index_single.t().tolist()
        pair_identity_list = []
        for src, dst in edge_pairs_list:
            pair_id = min(src, dst) // 2
            one_hot = [0.0] * 4
            if 0 <= pair_id < 4:
                one_hot[pair_id] = 1.0
            pair_identity_list.append(one_hot)
        pair_identity_single = torch.tensor(pair_identity_list, dtype=torch.float) # (num_edges, 4)

        # Detect participants
        all_participants = set()
        for cls in class_folders:
            cls_path = osp.join(self.raw_dir, cls)
            if not osp.isdir(cls_path): continue
            for f in os.listdir(cls_path):
                if f.endswith('.npy'):
                    p_name = f.split('_')[0]
                    all_participants.add(p_name)
        sorted_participants = sorted(list(all_participants))
        p_mapping = {p: i for i, p in enumerate(sorted_participants)}
        print(f"[*] Detected {len(sorted_participants)} participants: {sorted_participants}")

        for cls in class_folders:
            cls_path = osp.join(self.raw_dir, cls)
            if not osp.isdir(cls_path):
                continue

            label = class_to_label[cls]

            for file in tqdm(os.listdir(cls_path), desc=f'Processing {cls}'):
                if not file.endswith('.npy'):
                    continue

                p_name = file.split('_')[0]
                arr = np.load(osp.join(cls_path, file))  # (30, 8, 2)
                p_id = p_mapping[p_name]

                # 2. Edge Feature: Compact Delta Summary (4-D: mean/std of RSSI/Phase deltas)
                dynamic_edge_attrs = []
                for src, dst in edge_pairs_list:
                    delta_rssi = arr[:, dst, 0] - arr[:, src, 0]
                    delta_phase_raw = arr[:, dst, 1] - arr[:, src, 1]
                    delta_phase = (delta_phase_raw + np.pi) % (2 * np.pi) - np.pi
                    
                    summary = [
                        float(np.mean(delta_rssi)),
                        float(np.std(delta_rssi)),
                        float(np.mean(delta_phase)),
                        float(np.std(delta_phase))
                    ]
                    dynamic_edge_attrs.append(summary)
                
                dynamic_edge_attr_single = torch.tensor(dynamic_edge_attrs, dtype=torch.float)
                # Combine: (num_edges, 4) cat (num_edges, 4) -> (num_edges, 8)
                edge_attr_combined = torch.cat([pair_identity_single, dynamic_edge_attr_single], dim=1)

                # Node Features: each tag gets all 30 time steps × 2 features = 60 features
                node_features = []
                for tag in range(8):
                    tag_signal = arr[:, tag, :]      # (30, 2)
                    tag_feat = tag_signal.reshape(-1)  # (60,)
                    node_features.append(tag_feat)

                x = torch.tensor(node_features, dtype=torch.float)
                y = torch.tensor([label], dtype=torch.long)

                data = Data(
                    x=x,
                    edge_index=edge_index_single,
                    edge_attr=edge_attr_combined,
                    y=y,
                    participant=torch.tensor([p_id], dtype=torch.long)
                )

                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f"[*] Total samples processed: {len(data_list)}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
