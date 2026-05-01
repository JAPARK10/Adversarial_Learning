"""
lopo_train.py  —  Leave-One-Participant-Out with validation person.

For every combination of (test_person, val_person) where test != val:
    - Train on the other 14 participants
    - Validate on val_person (for early stopping)
    - Test on test_person (final accuracy)

Total runs: 16 x 15 = 240 combinations
Final result: mean and std of test accuracy across all 240 runs.

HOW TO RUN:
    conda activate GNNPlus
    cd /Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main
    python lopo_train.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from itertools import permutations

# ── HYPERPARAMETERS (Tuning Area) ──────────────────────────────────────────
NUM_EPOCHS       = 150
BATCH_SIZE       = 64
LR               = 0.001   
DIM_IN           = 120     
DIM_HIDDEN       = 128     
DROPOUT          = 0.2
ADV_LAMBDA       = 1.0     
ORTHO_WEIGHT     = 1.0     # Now dynamically modulated by certainty

# --- IMPROVEMENT TOGGLES (Ablation Monitoring) ---
USE_GAT          = True    # Graph Attention (v2)
USE_SUPCON       = True    # Supervised Contrastive Loss
USE_SCHEDULER    = True    # Cosine Annealing LR
# --------------------------------------------------
# Weight for Disentanglement loss
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_PT       = os.path.join(BASE_DIR, 'RFIDDataSet', 'processed', 'super_geometric_data.pt')
NUM_PARTICIPANTS = 16
NUM_GESTURES     = 22
RESULTS_FILE     = 'lopo_results.txt'
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────────────────────────


# ── Gradient Reversal Layer ───────────────────────────────────────────────────
class _GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GradientReversal(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return _GradRevFn.apply(x, self.lam)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning Loss."""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D], labels: [B]
        features = torch.nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Remove self-similarity from denominator
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(features.shape[0]).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # Mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

class UserDiscriminator(nn.Module):
    def __init__(self, in_dim, num_participants, use_grl=True, lam=1.0):
        super().__init__()
        self.use_grl = use_grl
        if use_grl:
            self.grl = GradientReversal(lam)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_participants),
        )

    def forward(self, x):
        if self.use_grl:
            x = self.grl(x)
        return self.mlp(x)


def get_adversarial_lambda(epoch, max_epochs):
    """Logistic warmup for the adversarial weight lambda."""
    p = float(epoch) / max_epochs
    return 2. / (1. + np.exp(-10. * p)) - 1.


# ── GCN Model ─────────────────────────────────────────────────────────────────
class GestureGCN(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_classes):
        super(GestureGCN, self).__init__()
        # Shared Pre-processing
        self.pre_mp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        # GNN Layers (Toggleable GAT vs GCN)
        if USE_GAT:
            self.conv1 = GATv2Conv(dim_hidden, dim_hidden, heads=4, concat=False)
            self.conv2 = GATv2Conv(dim_hidden, dim_hidden, heads=4, concat=False)
        else:
            self.conv1 = GCNConv(dim_hidden, dim_hidden)
            self.conv2 = GCNConv(dim_hidden, dim_hidden)
        
        # Public Branch (Gesture Essence)
        self.public_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU()
        )
        self.classifier = nn.Linear(dim_hidden, num_classes)
        
        # Private Branch (Person Specifics)
        self.private_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, NUM_PARTICIPANTS)
        )

    def forward(self, x, edge_index, batch, grl_lambda=1.0):
        x = self.pre_mp(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        # Disentangle
        z_pub = self.public_head(x)
        z_pri = self.private_head(x)

        # Gesture Prediction (Public)
        out_gesture = self.classifier(z_pub)

        # Identity Prediction (Private + GRL)
        z_pri_grl = GradientReversalLayer.apply(z_pri, grl_lambda)
        out_identity = self.discriminator(z_pri_grl)

        return out_gesture, out_identity, z_pub, z_pri


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_full_dataset():
    print(f'Loading from: {DATASET_PT}')
    data_store, slices = torch.load(DATASET_PT, weights_only=False)
    num_samples = slices['x'].shape[0] - 1
    dataset = []
    for i in range(num_samples):
        d = Data()
        s, e = slices['x'][i].item(), slices['x'][i+1].item()
        d.x = data_store.x[s:e].clone() 

        s, e = slices['edge_index'][i].item(), slices['edge_index'][i+1].item()
        d.edge_index = data_store.edge_index[:, s:e]

        s, e = slices['y'][i].item(), slices['y'][i+1].item()
        d.y = data_store.y[s:e]
        s, e = slices['p_y'][i].item(), slices['p_y'][i+1].item()
        d.p_y = data_store.p_y[s:e]
        dataset.append(d)
    return dataset


def split_three_way(dataset, test_pid, val_pid):
    """Split into train / val / test by participant ID."""
    print(f"    Splitting data (Test: p{test_pid+1}, Val: p{val_pid+1})...")
    train, val, test = [], [], []
    for d in dataset:
        pid = d.p_y.item()
        if pid == test_pid:
            test.append(d)
        elif pid == val_pid:
            val.append(d)
        else:
            train.append(d)
    return train, val, test


def evaluate(model, loader):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    all_true, all_pred, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred, _, _, _ = model(batch.x, batch.edge_index, batch.batch)
            true = batch.y.squeeze(-1)
            
            all_true.extend(true.cpu().numpy())
            all_pred.extend(pred.argmax(dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(pred, dim=1).cpu().numpy())

    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average='macro')
    
    # Identify failing classes
    errors_per_class = {}
    for t, p in zip(all_true, all_pred):
        if t != p:
            errors_per_class[t] = errors_per_class.get(t, 0) + 1
            
    return acc, f1, errors_per_class


def train_one_combination(train_data, val_data, test_data,
                          test_pid, val_pid, run_idx, total_runs):
    print(f"    Initializing model and loaders for run {run_idx}/{total_runs}...")
    model = GestureGCN(DIM_IN, DIM_HIDDEN, NUM_GESTURES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Cosine Annealing (Optional Improvement)
    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Filter Val/Test to only use "Original" samples (every 3rd sample in the Super-Dataset)
    val_data_orig = val_data[::3]
    test_data_orig = test_data[::3]

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data_orig,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data_orig,  batch_size=BATCH_SIZE, shuffle=False)

    # Use Label Smoothing to improve generalization
    criterion_gesture = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_identity = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss().to(DEVICE)

    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_f1       = 0.0
    best_errors   = {}

    for epoch in range(NUM_EPOCHS):
        model.train()
        print(f"    Epoch {epoch+1} starting...", end='\r')
        
        total_train_correct = 0
        total_train_samples = 0
        
        current_lam = get_adversarial_lambda(epoch, NUM_EPOCHS)

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out_g, out_p, z_pub, z_pri = model(batch.x, batch.edge_index, batch.batch, current_lam)
            
            # Monitoring Training Accuracy
            pred_g = out_g.argmax(dim=1)
            total_train_correct += (pred_g == batch.y.squeeze(-1)).sum().item()
            total_train_samples += batch.y.size(0)
            
            # 1. Main Gesture Loss
            loss_gesture = criterion_gesture(out_g, batch.y.squeeze(-1))
            
            # 2. SupCon Loss (Optional Improvement)
            if USE_SUPCON:
                loss_supcon = supcon_criterion(z_pub, batch.y.squeeze(-1))
                loss_gesture = 0.5 * loss_gesture + 0.5 * loss_supcon

            # 3. Adversarial Loss (DANN) - Per sample for dynamic weighting
            loss_adv_per_sample = torch.nn.functional.cross_entropy(out_p, batch.p_y.squeeze(-1), reduction='none')
            loss_adv = loss_adv_per_sample.mean()

            # 4. Discriminator Entropy (Monitoring "Confusion")
            probs_p = torch.softmax(out_p, dim=1)
            entropy = -torch.sum(probs_p * torch.log(probs_p + 1e-6), dim=1).mean()

            # 5. Dynamic Orthogonality Loss (Per-Sample)
            z_pub_n = torch.nn.functional.normalize(z_pub, p=2, dim=1)
            z_pri_n = torch.nn.functional.normalize(z_pri, p=2, dim=1)
            ortho_per_sample = (z_pub_n * z_pri_n).sum(dim=1).pow(2)
            certainty_weight = torch.exp(-loss_adv_per_sample)
            loss_ortho = (certainty_weight * ortho_per_sample).mean()

            loss = loss_gesture + (current_lam * loss_adv) + (ORTHO_WEIGHT * loss_ortho)
            
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()

        # Check accuracies
        train_acc = total_train_correct / total_train_samples
        val_acc, _, _ = evaluate(model, val_loader)
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1:03d}/{NUM_EPOCHS} | L: {loss.item():.4f} (G:{loss_gesture.item():.2f} Adv:{loss_adv.item():.2f}) | Ent: {entropy.item():.2f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}')

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc, best_f1, best_errors = evaluate(model, test_loader)

    return best_test_acc, best_f1, best_errors


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'\n{"="*50}')
    print(f' VERSION: IMPROVED (GAT:{USE_GAT}, SupCon:{USE_SUPCON}, Sch:{USE_SCHEDULER})')
    print(f' RUNNING ON: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f' GPU NAME:   {torch.cuda.get_device_name(0)}')
    print(f'{"="*50}\n')

    print('Loading dataset...')
    dataset = load_full_dataset()
    print(f'Total samples: {len(dataset)}')

    pids = sorted(set(d.p_y.item() for d in dataset))
    print(f'Participants found: {pids}')
    
    combinations = [(15, 1), (12, 15)]
    
    total_runs = len(combinations)
    print(f'Total combinations: {total_runs} (Ultra-Fast Mode)')

    all_acc, all_f1 = [], []

    for run_idx, (test_pid, val_pid) in enumerate(combinations, 1):
        train_data, val_data, test_data = split_three_way(
            dataset, test_pid, val_pid
        )

        if len(test_data) == 0 or len(val_data) == 0:
            print(f'  Skipping: empty split for test=p{test_pid+1} val=p{val_pid+1}')
            continue

        acc, f1, class_errors = train_one_combination(
            train_data, val_data, test_data,
            test_pid, val_pid, run_idx, total_runs
        )
        
        print(f"  [DONE] Acc: {acc:.4f} | F1: {f1:.4f}")
        # Print Top 3 Failing Gestures
        sorted_errors = sorted(class_errors.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top Errors: " + ", ".join([f"G{k+1}({v})" for k, v in sorted_errors[:3]]))
        
        all_acc.append(acc)
        all_f1.append(f1)

    # ── Final summary ────────────────────────────────────────────────────────
    mean_acc = np.mean(all_acc)
    std_acc  = np.std(all_acc)
    mean_f1  = np.mean(all_f1)
    mean_auc = np.mean(all_auc)

    print(f'\n{"="*60}')
    print(f'LOPO RESULTS  ({total_runs} combinations, 16 test x 15 val)')
    print(f'{"="*60}')
    print(f'  Mean Test Accuracy : {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'  Mean F1            : {mean_f1:.4f}')
    print(f'  Mean AUC           : {mean_auc:.4f}')
    print(f'{"="*60}')

    # Per-test-participant average
    print(f'\nPer-participant average (averaged over all 15 val choices):')
    for test_pid in range(NUM_PARTICIPANTS):
        indices = [i for i, (t, v) in enumerate(combinations) if t == test_pid]
        if indices:
            p_acc = np.mean([all_acc[i] for i in indices])
            print(f'  p{test_pid+1:02d}: {p_acc:.4f}')

    with open(RESULTS_FILE, 'w') as f:
        f.write('LOPO Cross-Validation Results (test+val split)\n')
        f.write(f'Adversarial lambda: DYNAMIC (Logistic Warmup)\n')
        f.write(f'Epochs per run: {NUM_EPOCHS}\n')
        f.write(f'Total combinations: {total_runs}\n\n')
        for (test_pid, val_pid), acc, f1, auc in zip(
                combinations, all_acc, all_f1, all_auc):
            f.write(f'test=p{test_pid+1:02d} val=p{val_pid+1:02d}: '
                    f'acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}\n')
        f.write(f'\nMean Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}\n')
        f.write(f'Mean F1       : {mean_f1:.4f}\n')
        f.write(f'Mean AUC      : {mean_auc:.4f}\n')

    print(f'\nResults saved to {RESULTS_FILE}')


if __name__ == '__main__':
    main()