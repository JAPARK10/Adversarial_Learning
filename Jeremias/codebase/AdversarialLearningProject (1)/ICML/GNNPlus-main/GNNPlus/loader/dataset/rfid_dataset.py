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
        print("RAW DIR EXISTS:", os.path.exists(self.raw_dir))
        print("RAW DIR CONTENT:", os.listdir(self.raw_dir))

        data_list = []

        # Find all participant folders
        participants = sorted([d for d in os.listdir(self.raw_dir) 
                             if os.path.isdir(osp.join(self.raw_dir, d))])
        
        participant_to_id = {p: i for i, p in enumerate(participants)}
        
        # We need a global mapping for exercises across all participants
        all_exercises = set()
        for p in participants:
            p_path = osp.join(self.raw_dir, p)
            ex_folders = [d for d in os.listdir(p_path) if os.path.isdir(osp.join(p_path, d))]
            all_exercises.update(ex_folders)
        
        exercise_to_label = {ex: i for i, ex in enumerate(sorted(list(all_exercises)))}

        print(f"[*] Found {len(participants)} participants and {len(exercise_to_label)} exercises.")

        # Fixed physical proximity edges
        edge_pairs = [
            (0, 1), (1, 0),
            (2, 3), (3, 2),
            (4, 5), (5, 4),
            (6, 7), (7, 6),
        ]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t()

        for p in participants:
            p_id = participant_to_id[p]
            p_path = osp.join(self.raw_dir, p)
            
            ex_folders = sorted([d for d in os.listdir(p_path) if os.path.isdir(osp.join(p_path, d))])
            
            for ex in ex_folders:
                ex_path = osp.join(p_path, ex)
                label = exercise_to_label[ex]

                for file in tqdm(os.listdir(ex_path), desc=f'Processing {p}/{ex}'):
                    if not file.endswith('.npy'):
                        continue

                    arr = np.load(osp.join(ex_path, file))  # (30, 8, 2)

                    # Build node features
                    node_features = []
                    for tag in range(8):
                        tag_signal = arr[:, tag, :]      # (30, 2)
                        tag_feat = tag_signal.reshape(-1)  # (60,)
                        node_features.append(tag_feat)

                    x = torch.tensor(node_features, dtype=torch.float)
                    y = torch.tensor([label], dtype=torch.long)
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
