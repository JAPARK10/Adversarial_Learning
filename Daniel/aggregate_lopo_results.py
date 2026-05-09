import json
import numpy as np

def aggregate_results(val_path, test_path, epochs_per_fold=150, num_folds=16):
    with open(val_path, 'r') as f:
        val_stats = [json.loads(line) for line in f if line.strip()]
    
    with open(test_path, 'r') as f:
        test_stats = [json.loads(line) for line in f if line.strip()]
    
    participants = ['DE', 'DS', 'FE', 'FS', 'HE', 'HS', 'JE', 'JS', 'KE', 'KS', 'LE', 'LS', 'NE', 'NS', 'PE', 'PS']
    
    results = []
    
    for i in range(num_folds):
        start = i * epochs_per_fold
        end = (i + 1) * epochs_per_fold
        
        fold_val = val_stats[start:end]
        fold_test = test_stats[start:end]
        
        # Find index of best validation accuracy
        best_idx = 0
        best_val_acc = -1
        for idx, stat in enumerate(fold_val):
            if stat['accuracy'] > best_val_acc:
                best_val_acc = stat['accuracy']
                best_idx = idx
        
        best_test_stat = fold_test[best_idx]
        
        test_participant = participants[i]
        val_participant = participants[(i + 1) % num_folds]
        
        results.append({
            'fold': i,
            'test_user': test_participant,
            'val_user': val_participant,
            'best_epoch': best_idx,
            'val_acc': best_val_acc,
            'test_acc': best_test_stat['accuracy'],
            'test_f1': best_test_stat['f1'],
            # Note: GraphGym might not save precision/recall in stats.json by default if not configured, 
            # but I can check if they exist.
            'test_auc': best_test_stat.get('auc', 0)
        })
        
    return results

if __name__ == "__main__":
    val_path = r'c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results\val\stats.json'
    test_path = r'c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\results\test\stats.json'
    
    res = aggregate_results(val_path, test_path)
    
    print("| Fold | Test User | Val User | Best Epoch | Test Acc | Test F1 | Val Acc |")
    print("|------|-----------|----------|------------|----------|---------|---------|")
    
    accs = []
    f1s = []
    
    for r in res:
        print(f"| {r['fold']:2d} | {r['test_user']:9s} | {r['val_user']:8s} | {r['best_epoch']:10d} | {r['test_acc']:.4f} | {r['test_f1']:.4f} | {r['val_acc']:.4f} |")
        accs.append(r['test_acc'])
        f1s.append(r['test_f1'])
    
    print(f"\nMean Accuracy: {np.mean(accs):.4f} \u00b1 {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} \u00b1 {np.std(f1s):.4f}")
    
    # Identify hardest/easiest
    hardest = res[np.argmin(accs)]
    easiest = res[np.argmax(accs)]
    
    print(f"\nHardest Participant: {hardest['test_user']} ({hardest['test_acc']:.4f})")
    print(f"Easiest Participant: {easiest['test_user']} ({easiest['test_acc']:.4f})")
