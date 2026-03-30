import torch
import torch_geometric
import numpy as np
from torch_geometric.data import Dataset

print(f"PyG version: {torch_geometric.__version__}")

class TestDataset(Dataset):
    def len(self):
        return 1
    def get(self, idx):
        return None

ds = TestDataset()
try:
    print("Testing numpy array [0]...")
    ds[np.array([0])]
    print("Numpy indexing works")
except Exception as e:
    print(f"Numpy indexing failed: {type(e).__name__}: {e}")

try:
    print("Testing numpy scalar np.int64(0)...")
    ds[np.int64(0)]
    print("Numpy scalar works")
except Exception as e:
    print(f"Numpy scalar failed: {type(e).__name__}: {e}")
