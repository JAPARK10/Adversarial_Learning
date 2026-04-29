"""
lopo_transformer.py  —  LOPO with Transformer temporal encoding.

Architecture:
    Per tag: (30, 2) → Transformer Encoder → 64-dim temporal embedding
    8 nodes × 64-dim → GCN × 3 → Mean pool → Label head + Adversarial head

HOW TO RUN:
    conda activate GNNPlus
    cd /Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main
    python lopo_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PT       = '/Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main/RFIDDataSet/processed/geometric_data_processed.pt'
NUM_PARTICIPANTS = 16
NUM_GESTURES     = 21
NUM_EPOCHS       = 150
BATCH_SIZE       = 32
LR               = 0.001
ADV_LAMBDA       = 0.5
DIM_HIDDEN       = 128
DIM_TEMPORAL     = 64    # transformer model dimension
TEMPORAL_HEADS   = 4     # attention heads (DIM_TEMPORAL must be divisible by this)
TEMPORAL_LAYERS  = 2     # transformer encoder layers
DROPOUT          = 0.2
RESULTS_FILE     = 'lopo_transformer_results.txt'
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


# ── Transformer Temporal Encoder ──────────────────────────────────────────────
class TemporalTransformerEncoder(nn.Module):
    """
    Encodes a single tag's time series (30, 2) into a fixed-size
    temporal embedding using self-attention.

    Steps:
        1. Linear projection: (30, 2) → (30, DIM_TEMPORAL)
        2. Add learned positional encoding
        3. Transformer encoder: (30, DIM_TEMPORAL) → (30, DIM_TEMPORAL)
        4. Mean pool over timesteps → (DIM_TEMPORAL,)
    """
    def __init__(self, input_dim=2, d_model=DIM_TEMPORAL,
                 nhead=TEMPORAL_HEADS, num_layers=TEMPORAL_LAYERS,
                 dropout=0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(30, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        """
        x: (N_nodes, 30, 2)
        returns: (N_nodes, DIM_TEMPORAL)
        """
        x = self.input_proj(x)
        positions = torch.arange(30, device=x.device)
        x = x + self.pos_embedding(positions)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x


# ── GCN Model with Transformer Temporal Encoding ─────────────────────────────
class GestureGCN(nn.Module):
    def __init__(self, dim_hidden=DIM_HIDDEN,
                 num_classes=NUM_GESTURES, dropout=DROPOUT):
        super().__init__()

        self.temporal_encoder = TemporalTransformerEncoder()

        self.pre_mp = nn.Linear(DIM_TEMPORAL, dim_hidden)

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
        x_flat     = data.x
        edge_index = data.edge_index
        batch      = data.batch

        x = x_flat.view(-1, 30, 2)
        x = self.temporal_encoder(x)
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


def split_by_participant(dataset, test_pid):
    train, test = [], []
    for d in dataset:
        pid = d.participant.item()
        if pid == test_pid:
            test.append(d)
        else:
            train.append(d)
    return train, test


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


def train_one_fold(train_data, test_data, fold):
    print(f'\n{"="*60}')
    print(f'  FOLD {fold+1:02d}/{NUM_PARTICIPANTS} — Test: p{fold+1:02d}')
    print(f'  Train: {len(train_data)} | Test: {len(test_data)}')
    print(f'{"="*60}')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    model     = GestureGCN()
    user_disc = UserDiscriminator(DIM_HIDDEN, NUM_PARTICIPANTS, ADV_LAMBDA)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(user_disc.parameters()),
        lr=LR, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

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
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{NUM_EPOCHS}')

    acc, f1, auc = evaluate(model, test_loader)
    print(f'  → Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}')
    return acc, f1, auc


def main():
    print('=== Transformer + GCN + Adversarial LOPO ===')
    print('Loading dataset...')
    dataset = load_full_dataset()
    print(f'Total samples: {len(dataset)}')

    all_acc, all_f1, all_auc = [], [], []

    for fold in range(NUM_PARTICIPANTS):
        train_data, test_data = split_by_participant(dataset, fold)
        if len(test_data) == 0:
            continue
        acc, f1, auc = train_one_fold(train_data, test_data, fold)
        all_acc.append(acc)
        all_f1.append(f1)
        all_auc.append(auc)

    mean_acc = np.mean(all_acc)
    std_acc  = np.std(all_acc)
    mean_f1  = np.mean(all_f1)
    mean_auc = np.mean(all_auc)

    print(f'\n{"="*60}')
    print(f'Transformer + GCN LOPO RESULTS')
    print(f'{"="*60}')
    for i, (acc, f1, auc) in enumerate(zip(all_acc, all_f1, all_auc)):
        print(f'  p{i+1:02d}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}')
    print(f'{"─"*60}')
    print(f'  Mean Accuracy : {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'  Mean F1       : {mean_f1:.4f}')
    print(f'  Mean AUC      : {mean_auc:.4f}')
    print(f'{"="*60}')

    with open(RESULTS_FILE, 'w') as f:
        f.write('Transformer + GCN + Adversarial LOPO Results\n')
        f.write(f'ADV_LAMBDA={ADV_LAMBDA}, EPOCHS={NUM_EPOCHS}\n\n')
        for i, (acc, f1, auc) in enumerate(zip(all_acc, all_f1, all_auc)):
            f.write(f'p{i+1:02d}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}\n')
        f.write(f'\nMean Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}\n')
        f.write(f'Mean F1       : {mean_f1:.4f}\n')
        f.write(f'Mean AUC      : {mean_auc:.4f}\n')

    print(f'Results saved to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
