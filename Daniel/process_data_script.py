import sys
import os
import traceback
import torch
import functools

# PyTorch 2.6+ changed torch.load to default to weights_only=True
# PyTorch Geometric objects cannot be pickled with this securely.
# So we monkeypatch torch.load to use weights_only=False globally just for this script run
original_load = torch.load
torch.load = functools.partial(original_load, weights_only=False)

sys.path.append(os.path.abspath(r'c:\Users\PC\Desktop\AdversarialLearningProject 1\ICML\GNNPlus-main\GNNPlus'))
from loader.dataset.rfid_dataset import RFIDDataset

if __name__ == '__main__':
    root_path = r'c:\Users\PC\Desktop\AdversarialLearningProject 1\ICML\GNNPlus-main\datasets'
    print(f"Instantiating RFIDDataset with root path: {root_path}")
    
    try:
        dataset = RFIDDataset(root=root_path)
        print("Dataset has been processed and loaded successfully!")
        print(f"Total number of graphs: {len(dataset)}")
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")
        if len(dataset) > 0:
            print(f"First graph: {dataset[0]}")
    except Exception as e:
        print(f"Failed to instantiate dataset: {e}")
        traceback.print_exc()
