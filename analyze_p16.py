import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def analyze():
    data_path = r"c:\Users\jerem\Desktop\Workspace_VSCode\CoDaS\Adversarial_Learning\Farhan\ICML\GNNPlus-main\RFIDDataSet\processed\super_geometric_data.pt"
    print(f"Loading dataset from {data_path}...")
    data_store, slices = torch.load(data_path, weights_only=False)
    
    leaderboard = []
    
    # Analyze all 16 participants
    for pid in range(16):
        indices = (data_store.p_y == pid).nonzero(as_tuple=True)[0]
        if len(indices) == 0: continue
        
        # Sample 50 gestures per person for a quick check
        subset = indices[::max(1, len(indices)//50)]
        
        person_vars = []
        person_mags = []
        
        for idx in subset:
            s, e = slices['x'][idx], slices['x'][idx+1]
            x = data_store.x[s:e]
            rssi = x[:, 0:30]
            person_mags.append(rssi.abs().mean().item())
            person_vars.append(rssi.var(dim=1).mean().item())
            
        leaderboard.append({
            'pid': pid + 1,
            'mag': np.mean(person_mags),
            'speed': np.mean(person_vars)
        })
    
    # Calculate Global Average Speed
    avg_speed = np.mean([p['speed'] for p in leaderboard])
    
    # Sort by speed (Fastest to Slowest)
    leaderboard.sort(key=lambda x: x['speed'], reverse=True)
    
    print("\n" + "="*50)
    print(f"{'PID':<10} | {'MAGNITUDE':<15} | {'SPEED (VAR)':<15} | {'DIFF TO AVG'}")
    print("-" * 50)
    for p in leaderboard:
        diff = (p['speed'] / avg_speed - 1) * 100
        print(f"Person {p['pid']:02d} | {p['mag']:<15.4f} | {p['speed']:<15.4f} | {diff:>+6.1f}%")
    print("="*50)
    
    print("\nOBSERVATION:")
    fastest = leaderboard[0]
    slowest = leaderboard[-1]
    print(f"  * Fastest Person is P{fastest['pid']:02d} ({fastest['speed']/avg_speed:.1f}x average)")
    print(f"  * Slowest Person is P{slowest['pid']:02d} ({slowest['speed']/avg_speed:.1f}x average)")
    
    if fastest['pid'] == 16:
        print("\nCONFIRMED: Subject 16 is indeed the speed outlier. Temporal Normalization is required.")

if __name__ == "__main__":
    analyze()
