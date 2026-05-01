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
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from itertools import permutations

# ── HYPERPARAMETERS (Tuning Area) ──────────────────────────────────────────
NUM_EPOCHS       = 150
BATCH_SIZE       = 64
LR               = 0.001   # Try 0.01 or 0.005
DIM_IN           = 60
DIM_HIDDEN       = 128     # Reduced for better generalization
DROPOUT          = 0.2
ADV_LAMBDA       = 1.0     # Increased identity scrubbing
ORTHO_WEIGHT     = 0.001   # Weight for Disentanglement loss
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_PT       = os.path.join(BASE_DIR, 'RFIDDataSet', 'processed', 'geometric_data_processed.pt')
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
    def __init__(self, dim_in=DIM_IN, dim_hidden=DIM_HIDDEN,
                 num_classes=NUM_GESTURES, dropout=DROPOUT):
        super().__init__()
        self.pre_mp = nn.Linear(dim_in, dim_hidden)
        # 3-layer GCN for more representational room
        self.convs = nn.ModuleList([
            GCNConv(dim_hidden, dim_hidden) for _ in range(3)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(dim_hidden) for _ in range(3)
        ])
        self.ff1 = nn.ModuleList([
            nn.Linear(dim_hidden, dim_hidden * 2) for _ in range(3)
        ])
        self.ff2 = nn.ModuleList([
            nn.Linear(dim_hidden * 2, dim_hidden) for _ in range(3)
        ])
        self.dropout = dropout
        self.post_mp = nn.Sequential(
            nn.Linear(dim_hidden // 2, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.pre_mp(x))
        for conv, bn, ff1, ff2 in zip(self.convs, self.bns,
                                       self.ff1, self.ff2):
            identity = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = ff1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = ff2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity
        
        # Using Add pooling (preserves structural energy)
        graph_embed = global_add_pool(x, batch)
        
        # [Split] Disentangle into Public (0:64) and Private (64:128) branches
        z_pub = graph_embed[:, :graph_embed.shape[1] // 2]
        z_priv = graph_embed[:, graph_embed.shape[1] // 2:]
        
        out = self.post_mp(z_pub)
        return out, z_pub, z_priv


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_full_dataset():
    print(f'Loading from: {DATASET_PT}')
    data_store, slices = torch.load(DATASET_PT, weights_only=False)
    num_samples = slices['x'].shape[0] - 1
    dataset = []
    for i in range(num_samples):
        d = Data()
        s, e = slices['x'][i].item(), slices['x'][i+1].item()
        d.x = data_store.x[s:e].clone() # Clone to avoid modifying the original data_store

        # [Normalization] Standardize features to zero mean and unit variance per sample
        # This removes the absolute magnitude shortcut for identity detection
        x_mean = d.x.mean()
        x_std = d.x.std() + 1e-7
        d.x = (d.x - x_mean) / x_std

        # [Graph Structure] Create a Fully Connected graph for 8 tags
        # This allows the GNN to learn any cross-tag spatial relationship
        num_nodes = 8
        adj = torch.ones((num_nodes, num_nodes))
        d.edge_index = adj.nonzero().t().contiguous()

        s, e = slices['y'][i].item(), slices['y'][i+1].item()
        d.y = data_store.y[s:e]
        s, e = slices['p_y'][i].item(), slices['p_y'][i+1].item()
        d.p_y = data_store.p_y[s:e]
        dataset.append(d)
    return dataset


def split_three_way(dataset, test_pid, val_pid):
    """Split into train / val / test by participant ID."""
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
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred, _, _ = model(batch)
            true = batch.y.squeeze(-1)
            probs = torch.softmax(pred, dim=1)
            all_true.extend(true.cpu().numpy())
            all_pred.extend(pred.argmax(dim=1).cpu().numpy())
            all_prob.extend(probs.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(all_true, all_prob,
                            multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0
    return acc, f1, auc


def train_one_combination(train_data, val_data, test_data,
                          test_pid, val_pid, run_idx, total_runs):
    model     = GestureGCN().to(DEVICE)
    # Public Disc (with GRL) to scrub identity
    user_disc_pub = UserDiscriminator(DIM_HIDDEN // 2, NUM_PARTICIPANTS, use_grl=True).to(DEVICE)
    # Private Disc (no GRL) to attract identity
    user_disc_priv = UserDiscriminator(DIM_HIDDEN // 2, NUM_PARTICIPANTS, use_grl=False).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(user_disc_pub.parameters()) + list(user_disc_priv.parameters()),
        lr=LR, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_f1       = 0.0
    best_auc      = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        user_disc_pub.train()
        user_disc_priv.train()
        
        # Update dynamic lambda for this epoch
        current_lam = get_adversarial_lambda(epoch, NUM_EPOCHS)
        user_disc_pub.grl.lam = current_lam

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            pred, z_pub, z_priv = model(batch)
            gesture_labels = batch.y.squeeze(-1)
            participant_labels = batch.p_y.squeeze(-1)

            # 1. Gesture Loss
            gesture_loss = F.cross_entropy(pred, gesture_labels)

            # 2. Public Branch: Identity Scrubbing (DANN approach)
            # The Discriminator tries to IDENTIFY (CrossEntropy).
            # The GRL flips the gradient for the GNN to HIDE.
            user_logits_pub = user_disc_pub(z_pub)
            adv_loss = F.cross_entropy(user_logits_pub, participant_labels)

            # 3. Private Branch: Identity Attraction
            user_logits_priv = user_disc_priv(z_priv)
            priv_loss = F.cross_entropy(user_logits_priv, participant_labels)

            # 4. Orthogonality Loss (Full Cross-Correlation)
            # We want to ensure that NO feature in z_pub correlates with ANY feature in z_priv
            # Across the whole batch: (Z_pub.T @ Z_priv) should be zero matrix
            # Subtract means to get covariance
            z_pub_cent = z_pub - z_pub.mean(dim=0, keepdim=True)
            z_priv_cent = z_priv - z_priv.mean(dim=0, keepdim=True)
            corr_matrix = torch.matmul(z_pub_cent.t(), z_priv_cent)
            ortho_loss = torch.norm(corr_matrix, p='fro') # Frobenius norm of the cross-correlation

            # Total Loss
            loss = gesture_loss + (current_lam * adv_loss) + priv_loss + (ORTHO_WEIGHT * ortho_loss)
            
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Live Progress update every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1:03d}/{NUM_EPOCHS} | L: {loss.item():.4f} (G: {gesture_loss.item():.4f}, Adv: {adv_loss.item():.4f}, Ortho: {ortho_loss.item():.4f}) | Lam: {current_lam:.3f}')

        # Check validation accuracy each epoch
        val_acc, _, _ = evaluate(model, val_loader)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Evaluate test at best val epoch
            best_test_acc, best_f1, best_auc = evaluate(model, test_loader)

    print(f'  [{run_idx:03d}/{total_runs}] '
          f'test=p{test_pid+1:02d} val=p{val_pid+1:02d} | '
          f'best_val={best_val_acc:.4f} | '
          f'test_acc={best_test_acc:.4f}')

    return best_test_acc, best_f1, best_auc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'\n{"="*50}')
    print(f' VERSION: IMPROVED (Disentangled + DANN + Normalization)')
    print(f' RUNNING ON: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f' GPU NAME:   {torch.cuda.get_device_name(0)}')
    print(f'{"="*50}\n')

    print('Loading dataset...')
    dataset = load_full_dataset()
    print(f'Total samples: {len(dataset)}')

    pids = sorted(set(d.p_y.item() for d in dataset))
    print(f'Participants found: {pids}')

    # --- FULL LOPO (240 runs) ---
    # combinations = [(t, v) for t in range(NUM_PARTICIPANTS) for v in range(NUM_PARTICIPANTS) if t != v]
    
    # --- PARTIAL LOPO (16 runs) ---
    # combinations = [(i, (i + 1) % NUM_PARTICIPANTS) for i in range(NUM_PARTICIPANTS)]
    
    # Ultra-Fast Iteration: 2 specific hardcoded pairs
    combinations = [(15, 1), (12, 15)]
    
    total_runs = len(combinations)
    print(f'Total combinations: {total_runs} (Ultra-Fast Mode)')

    all_acc, all_f1, all_auc = [], [], []

    for run_idx, (test_pid, val_pid) in enumerate(combinations, 1):
        train_data, val_data, test_data = split_three_way(
            dataset, test_pid, val_pid
        )

        if len(test_data) == 0 or len(val_data) == 0:
            print(f'  Skipping: empty split for test=p{test_pid+1} val=p{val_pid+1}')
            continue

        acc, f1, auc = train_one_combination(
            train_data, val_data, test_data,
            test_pid, val_pid, run_idx, total_runs
        )
        all_acc.append(acc)
        all_f1.append(f1)
        all_auc.append(auc)

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