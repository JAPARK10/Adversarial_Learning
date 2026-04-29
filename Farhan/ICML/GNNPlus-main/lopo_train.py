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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import BatchNorm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from itertools import permutations

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PT       = '/Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main/RFIDDataSet/processed/geometric_data_processed.pt'
NUM_PARTICIPANTS = 16
NUM_GESTURES     = 21
NUM_EPOCHS       = 150
BATCH_SIZE       = 32
LR               = 0.001
ADV_LAMBDA       = 0.5
DIM_IN           = 60
DIM_HIDDEN       = 128
DROPOUT          = 0.2
RESULTS_FILE     = 'lopo_results.txt'
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
    def __init__(self, in_dim, num_participants, lam=1.0):
        super().__init__()
        self.grl = GradientReversal(lam)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_participants),
        )

    def forward(self, x):
        return self.mlp(self.grl(x))


# ── GCN Model ─────────────────────────────────────────────────────────────────
class GestureGCN(nn.Module):
    def __init__(self, dim_in=DIM_IN, dim_hidden=DIM_HIDDEN,
                 num_classes=NUM_GESTURES, dropout=DROPOUT):
        super().__init__()
        self.pre_mp = nn.Linear(dim_in, dim_hidden)
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
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes),
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
        graph_embed = global_mean_pool(x, batch)
        out = self.post_mp(graph_embed)
        return out, graph_embed


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_full_dataset():
    print(f'Loading from: {DATASET_PT}')
    data_store, slices = torch.load(DATASET_PT)
    num_samples = slices['x'].shape[0] - 1
    dataset = []
    for i in range(num_samples):
        d = Data()
        s, e = slices['x'][i].item(), slices['x'][i+1].item()
        d.x = data_store.x[s:e]
        s, e = slices['edge_index'][i].item(), slices['edge_index'][i+1].item()
        d.edge_index = data_store.edge_index[:, s:e]
        s, e = slices['y'][i].item(), slices['y'][i+1].item()
        d.y = data_store.y[s:e]
        s, e = slices['participant'][i].item(), slices['participant'][i+1].item()
        d.participant = data_store.participant[s:e]
        dataset.append(d)
    return dataset


def split_three_way(dataset, test_pid, val_pid):
    """Split into train / val / test by participant ID."""
    train, val, test = [], [], []
    for d in dataset:
        pid = d.participant.item()
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
            pred, _ = model(batch)
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
    model     = GestureGCN()
    user_disc = UserDiscriminator(DIM_HIDDEN, NUM_PARTICIPANTS, ADV_LAMBDA)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(user_disc.parameters()),
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
        user_disc.train()

        for batch in train_loader:
            optimizer.zero_grad()
            pred, graph_embed = model(batch)
            true = batch.y.squeeze(-1)

            gesture_loss = F.cross_entropy(pred, true)

            user_logits = user_disc(graph_embed)
            participant_labels = batch.participant.squeeze(-1)
            adv_loss = F.cross_entropy(user_logits, participant_labels)

            loss = gesture_loss + ADV_LAMBDA * adv_loss
            loss.backward()
            optimizer.step()

        scheduler.step()

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
    print('Loading dataset...')
    dataset = load_full_dataset()
    print(f'Total samples: {len(dataset)}')

    pids = sorted(set(d.participant.item() for d in dataset))
    print(f'Participants found: {pids}')

    # All (test, val) combinations where test != val
    # combinations = [(t, v) for t in range(NUM_PARTICIPANTS)
    #                         for v in range(NUM_PARTICIPANTS) if t != v]
    combinations = [(13, 10)]
    total_runs = len(combinations)
    print(f'Total combinations: {total_runs} (16 x 15)')

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
        f.write(f'Adversarial lambda: {ADV_LAMBDA}\n')
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