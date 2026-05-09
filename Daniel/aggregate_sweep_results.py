import json
import os
import numpy as np

def get_best_stats(out_dir, epochs_per_fold=150, num_folds=16):
    val_path = os.path.join(out_dir, 'val', 'stats.json')
    test_path = os.path.join(out_dir, 'test', 'stats.json')
    
    if not os.path.exists(val_path) or not os.path.exists(test_path):
        return None
        
    with open(val_path, 'r') as f:
        val_stats = [json.loads(line) for line in f if line.strip()]
    with open(test_path, 'r') as f:
        test_stats = [json.loads(line) for line in f if line.strip()]
        
    results = []
    for i in range(num_folds):
        start = i * epochs_per_fold
        end = (i + 1) * epochs_per_fold
        
        fold_val = val_stats[start:end]
        fold_test = test_stats[start:end]
        
        if not fold_val: continue
        
        # Best epoch by validation accuracy
        best_idx = 0
        best_val_acc = -1
        for idx, stat in enumerate(fold_val):
            if stat.get('accuracy', 0) > best_val_acc:
                best_val_acc = stat['accuracy']
                best_idx = idx
        
        if best_idx < len(fold_test):
            results.append(fold_test[best_idx])
        else:
            results.append(None)
            
    return results

def aggregate():
    root = r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main"
    participants = ['DE', 'DS', 'FE', 'FS', 'HE', 'HS', 'JE', 'JS', 'KE', 'KS', 'LE', 'LS', 'NE', 'NS', 'PE', 'PS']
    
    experiment_dirs = {
        'Baseline': r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results_baseline",
        'Baseline OptC': r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results_lopo_baseline_optC",
        'Baseline OptD': r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results_lopo_baseline_optD",
        'Lambda=0.1': r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results_lopo_adv_lambda_0.1",
        'Lambda=0.3': r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results_lopo_adv_lambda_0.3",
        'Lambda=0.5': r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results_lopo_adv_lambda_0.5",
        'Lambda=1.0': r"c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results_lopo_adv_lambda_1.0",
    }
    
    all_data = {}
    for name, path in experiment_dirs.items():
        data = get_best_stats(path)
        if data:
            all_data[name] = data

    if not all_data:
        print("No results found yet.")
        return

    # Create Table
    header = "| Fold | Test User | Baseline | OptC | OptD | λ=0.1 | λ=0.3 | λ=0.5 | λ=1.0 |"
    sep    = "|------|-----------|----------|------|------|-------|-------|-------|-------|"
    print(header)
    print(sep)
    
    column_keys = ['Baseline', 'Baseline OptC', 'Baseline OptD', 'Lambda=0.1', 'Lambda=0.3', 'Lambda=0.5', 'Lambda=1.0']
    
    fold_accs = {k: [] for k in column_keys}
    fold_f1s = {k: [] for k in column_keys}
    
    for i in range(16):
        row = f"| {i:2d} | {participants[i]:9s} |"
        for k in column_keys:
            if k in all_data and i < len(all_data[k]) and all_data[k][i]:
                acc = all_data[k][i]['accuracy']
                row += f" {acc*100:6.2f}% |"
                fold_accs[k].append(acc)
                fold_f1s[k].append(all_data[k][i].get('f1', 0))
            else:
                row += "   -    |"
        print(row)
        
    print(sep)
    
    # Statistical summary
    acc_row = "| Mean Accuracy | - |"
    for k in column_keys:
        if fold_accs[k]:
            m, s = np.mean(fold_accs[k]), np.std(fold_accs[k])
            acc_row += f" {m*100:4.1f}±{s*100:4.1f} |"
        else:
            acc_row += "   -    |"
    print(acc_row)
    
    f1_row = "| Mean F1 | - |"
    for k in column_keys:
        if fold_f1s[k]:
            m, s = np.mean(fold_f1s[k]), np.std(fold_f1s[k])
            f1_row += f" {m:.4f}±{s:.4f} |"
        else:
            f1_row += "   -    |"
    print(f1_row)

if __name__ == "__main__":
    aggregate()
