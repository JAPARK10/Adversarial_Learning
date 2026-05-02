import torch
import numpy as np
import os
import re

def analyze_speed_vs_performance(dataset_path, results_path):
    print("="*60)
    print(" GESTURE EXECUTION SPEED VS. PERFORMANCE ANALYSIS")
    print("="*60)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return
    
    # 1. Load Dataset
    print(f"Loading dataset: {dataset_path}...")
    loaded = torch.load(dataset_path, weights_only=False)
    if isinstance(loaded, tuple):
        data_obj, slices = loaded
    else:
        data_obj = loaded
        slices = None

    # 2. Calculate Speed per Participant
    # Speed proxy: Mean absolute delta (RSSI_delta and Phase_delta)
    # Features are [RSSI(30), Phase(30), RSSI_Delta(30), Phase_Delta(30)]
    # Deltas are indices 60 to 120.
    
    p_y = data_obj.p_y
    unique_pids = torch.unique(p_y).tolist()
    
    speed_stats = {}
    
    print("Analyzing temporal dynamics per participant...")
    for pid in sorted(unique_pids):
        indices = (p_y == pid).nonzero(as_tuple=True)[0]
        
        subject_deltas = []
        for idx in indices:
            start = idx * 8
            end = (idx + 1) * 8
            # Extract only the delta features (60:120)
            deltas = data_obj.x[start:end, 60:120]
            subject_deltas.append(deltas.abs().mean())
        
        avg_speed = torch.stack(subject_deltas).mean().item()
        speed_stats[pid] = avg_speed

    # 3. Parse Performance Results
    performance_stats = {}
    if os.path.exists(results_path):
        print(f"Parsing results: {results_path}...")
        with open(results_path, 'r') as f:
            content = f.read()
            
        # Pattern: test=p01 val=p02: acc=0.7548
        # Note: We aggregate all test runs for a specific test_person
        for pid in unique_pids:
            p_str = f"p{pid+1:02d}"
            # Find all accuracies where this is the test subject
            accs = re.findall(rf"test={p_str} .*acc=([\d.]+)", content)
            if accs:
                avg_acc = np.mean([float(a) for a in accs])
                performance_stats[pid] = avg_acc
    else:
        print(f"Warning: {results_path} not found. Performance correlation will be skipped.")

    # 4. Display Results
    print("\n" + "-"*65)
    print(f"{'PID':<6} | {'Speed (Avg Delta)':<20} | {'Avg Accuracy':<15} | {'Notes'}")
    print("-"*65)
    
    all_speeds = []
    all_accs = []
    
    for pid in sorted(unique_pids):
        speed = speed_stats[pid]
        acc = performance_stats.get(pid, 0.0)
        
        note = ""
        if pid == 15: note = "<-- Problematic Subject"
        if acc > 0.85: note = "<-- High Performer"
        
        print(f"p{pid+1:02d}    | {speed:<20.4f} | {acc:<15.4f} | {note}")
        
        if acc > 0:
            all_speeds.append(speed)
            all_accs.append(acc)

    # 5. Calculate Correlation
    if len(all_accs) > 1:
        correlation = np.corrcoef(all_speeds, all_accs)[0, 1]
        print("-"*65)
        print(f"Correlation (Speed vs. Accuracy): {correlation:.4f}")
        
        if correlation > 0.5:
            print("RESULT: Strong POSITIVE correlation. Faster gestures = Higher accuracy.")
        elif correlation < -0.5:
            print("RESULT: Strong NEGATIVE correlation. Slower gestures = Higher accuracy.")
        else:
            print("RESULT: Weak correlation.")
    
    print("="*60)

if __name__ == "__main__":
    # Check current directory and adjust paths
    dataset = "RFIDDataSet/processed/super_geometric_data.pt"
    results = "lopo_results.txt"
    
    if not os.path.exists(dataset):
        # Try parent dir or sibling dir if needed
        dataset = os.path.join("..", dataset)
    
    analyze_speed_vs_performance(dataset, results)
