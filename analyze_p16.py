import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def analyze():
    # Portable path calculation
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "Farhan", "ICML", "GNNPlus-main", "RFIDDataSet", "processed", "super_geometric_data.pt")
    
    if not os.path.exists(data_path):
        # Fallback for server structure if different
        data_path = "/root/Adversarial_Learning/Farhan/ICML/GNNPlus-main/RFIDDataSet/processed/super_geometric_data.pt"

    print(f"Loading dataset from {data_path}...")
    data_store, slices = torch.load(data_path, weights_only=False)
    
    leaderboard = []
    
    # Analyze all 16 participants
    for pid in range(16):
        indices = (data_store.p_y == pid).nonzero(as_tuple=True)[0]
        if len(indices) == 0: continue
        
        # Sample 50 gestures per person for a quick check
        subset = indices[::max(1, len(indices)//50)]
        
        person_speed = []
        person_mags = []
        person_noise = []
        person_symmetry = [] # Distribution across 8 sensors
        
        for idx in subset:
            s, e = slices['x'][idx], slices['x'][idx+1]
            x = data_store.x[s:e] # [8, 120]
            
            # Features: 0-30:RSSI, 30-60:Phase, 60-90:RSSI_Delta, 90-120:Phase_Delta
            rssi = x[:, 0:30]
            phase = x[:, 30:60]
            deltas = x[:, 60:120]
            
            # 1. Speed (Already defined as Delta Magnitude)
            person_speed.append(deltas.abs().mean().item())
            
            # 2. Magnitude (Signal Strength)
            person_mags.append(rssi.abs().mean().item())
            
            # 3. Noise (High frequency variance)
            # Calculated as the variance of the deltas (acceleration)
            person_noise.append(deltas.var().item())
            
            # 4. Symmetry (Are they standing off-center?)
            # Measured as the Coefficient of Variation between the 8 sensors' magnitudes
            sensor_mags = rssi.abs().mean(dim=1) # [8]
            cv = sensor_mags.std() / (sensor_mags.mean() + 1e-7)
            person_symmetry.append(cv.item())
            
        leaderboard.append({
            'pid': pid + 1,
            'mag': np.mean(person_mags),
            'speed': np.mean(person_speed),
            'noise': np.mean(person_noise),
            'bias': np.mean(person_symmetry)
        })
    
    # Calculate Global Averages
    avg_speed = np.mean([p['speed'] for p in leaderboard])
    avg_noise = np.mean([p['noise'] for p in leaderboard])
    avg_bias  = np.mean([p['bias'] for p in leaderboard])
    
    # Sort by speed (Fastest to Slowest)
    leaderboard.sort(key=lambda x: x['speed'], reverse=True)
    
    print("\n" + "="*95)
    print(f"{'PID':<8} | {'SPEED (Δ)':<12} | {'NOISE':<12} | {'SPATIAL BIAS':<15} | {'CHARACTERISTIC'}")
    print("-" * 95)
    for p in leaderboard:
        # Determine labels
        traits = []
        if p['speed'] > avg_speed * 1.2: traits.append("⚡ FAST")
        if p['speed'] < avg_speed * 0.8: traits.append("🐢 SLOW")
        if p['noise'] > avg_noise * 1.5: traits.append("🔊 NOISY")
        if p['bias']  > avg_bias  * 1.5: traits.append("📐 OFF-CENTER")
        if not traits: traits.append("⚖️ BALANCED")
        
        trait_str = ", ".join(traits)
        print(f"P{p['pid']:02d}      | {p['speed']:<12.4f} | {p['noise']:<12.4f} | {p['bias']:<15.4f} | {trait_str}")
    print("="*95)
    
    print("\nDEEP SCAN OBSERVATIONS:")
    # Check for Subject 16 specifically
    p16 = next((p for p in leaderboard if p['pid'] == 16), None)
    if p16:
        print(f"  * Subject 16: Speed={p16['speed']/avg_speed:.1f}x, Noise={p16['noise']/avg_noise:.1f}x, Bias={p16['bias']/avg_bias:.1f}x")
        if p16['noise'] > avg_noise * 1.5:
            print("    ⚠️ ALERT: P16 has high signal noise. Adversarial branch needs higher lambda to ignore this.")
        if p16['bias'] > avg_bias * 1.5:
            print("    ⚠️ ALERT: P16 has high spatial bias (Off-center). GAT Attention is critical here.")

if __name__ == "__main__":
    analyze()
