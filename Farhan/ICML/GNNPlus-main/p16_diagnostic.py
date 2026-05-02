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
    
    # Flatten N and 8 to calculate stats over all nodes/samples
    flat_features = all_features.view(-1, all_features.size(-1))
    
    mean_val = flat_features.mean(dim=0)
    std_val = flat_features.std(dim=0)
    max_val = flat_features.max(dim=0)[0]
    min_val = flat_features.min(dim=0)[0]
    
    print(f"  Feature Mean (avg): {mean_val.mean().item():.4f}")
    print(f"  Feature Std (avg):  {std_val.mean().item():.4f}")
    print(f"  Avg Dynamic Range: {(max_val - min_val).mean().item():.4f}")
    
    # Analyze Temporal Variance (Speed indicator)
    # Variance across the time/feature dimension per node/sample
    temporal_variance = all_features.std(dim=2).mean() 
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
