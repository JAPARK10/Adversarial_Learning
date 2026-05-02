import torch
import numpy as np
import os

def analyze_subject(dataset_path, pid):
    print(f"\nAnalyzing Subject p{pid+1}...")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return None
    
    # Load collated data
    loaded = torch.load(dataset_path, weights_only=False)
    if isinstance(loaded, tuple):
        data_obj, slices = loaded
    else:
        data_obj = loaded
        slices = None

    # Filter indices for the specific participant
    p_y = data_obj.p_y
    indices = (p_y == pid).nonzero(as_tuple=True)[0]
    print(f"Total samples found: {len(indices)}")
    
    if len(indices) == 0:
        return None

    # Extract features for these indices
    # Each sample has 8 nodes, so features are at [index*8 : (index+1)*8]
    subject_features = []
    for idx in indices:
        start = idx * 8
        end = (idx + 1) * 8
        subject_features.append(data_obj.x[start:end])
    
    all_features = torch.stack(subject_features) # [N, 8, features]
    
    mean_val = all_features.mean(dim=(0,1))
    std_val = all_features.std(dim=(0,1))
    max_val = all_features.max(dim=(0,1))[0]
    min_val = all_features.min(dim=(0,1))[0]
    
    print(f"  Feature Mean: {mean_val.numpy()}")
    print(f"  Feature Std:  {std_val.numpy()}")
    print(f"  Dynamic Range: {(max_val - min_val).numpy()}")
    
    # Analyze Temporal Variance (Speed indicator)
    # Features are flattened [8 sensors * T features] or [8 sensors, T]
    # In our case, x is [8, T] usually.
    temporal_variance = all_features.std(dim=2).mean() # Variance across the time/feature dimension
    print(f"  Avg Temporal Variance: {temporal_variance.item():.4f}")

    return {
        'mean': mean_val,
        'std': std_val,
        'var': temporal_variance
    }

if __name__ == "__main__":
    # Relative path for the server
    path = "RFIDDataSet/processed/super_geometric_data.pt"
    
    # Compare p16 (Problematic) vs p10 (Perfect)
    stats_16 = analyze_subject(path, 15) 
    if stats_16:
        stats_10 = analyze_subject(path, 9)
        
        if stats_10:
            print("\n" + "="*30)
            print("COMPARISON: p16 vs p10")
            print("="*30)
            var_diff = (stats_16['var'] / stats_10['var'] - 1) * 100
            print(f"p16 is {var_diff:.1f}% more/less dynamic than p10")
