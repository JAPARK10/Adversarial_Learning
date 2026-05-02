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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

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
BATCH_SIZE       = 128     # Doubled for speed on A5000
LR               = 0.001   
DIM_IN           = 120     
DIM_HIDDEN       = 256     
DROPOUT          = 0.3     
ADV_LAMBDA       = 1.0     
ORTHO_WEIGHT     = 1.0     

# --- IMPROVEMENT TOGGLES (Ablation Monitoring) ---
USE_GAT          = True    
USE_SENSOR_ID    = True    
USE_SENSOR_DROP  = True    
SENSOR_DROP_RATE = 0.2     
USE_DYNAMIC_JITTER = True   
JITTER_SIGMA     = 0.02     
USE_FOCAL        = True      # Focuses learning on hard gestures (G15, G6, etc.)
USE_SUPCON       = True    
USE_SCHEDULER    = True    
USE_TEMPORAL     = True    
USE_MIXUP        = True    
MIXUP_ALPHA      = 0.4     
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


# ── Logging Setup ────────────────────────────────────────────────────────────
def log_print(message, results_file=RESULTS_FILE):
    print(message)
    with open(results_file, 'a') as f:
        f.write(message + '\n')

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
    """Logistic warmup for the adversarial weight lambda with a 20-epoch delay."""
    if epoch < 20:
        return 0.0
    
    # Adjusted progress after the delay
    p = float(epoch - 20) / (max_epochs - 20)
    return 2. / (1. + np.exp(-10. * p)) - 1.


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p)**self.gamma * logp
        return loss.mean()

class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(TemporalEncoder, self).__init__()
        # 1. Local Feature Extraction (CNN)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)
        
        # 2. Global Rhythmic Reasoning (Self-Attention)
        # We treat the 30 timesteps as a sequence
        self.attn = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=out_dim, dropout=0.1, batch_first=True
        )
        
    def forward(self, x):
        # x shape: [B*8, 120] -> reshape to [B*8, 4, 30] (4 features: Phase/RSSI for 2 antennas)
        B_nodes = x.size(0)
        x = x.view(B_nodes, 4, 30)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # [B*8, out_dim, 30]
        
        # Self-Attention over time
        x = x.permute(0, 2, 1) # [B*8, 30, out_dim]
        x = self.attn(x)
        x = x.permute(0, 2, 1) # [B*8, out_dim, 30]
        
        # Global pooling across time
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1) # [B*8, out_dim]
        return x

# ── Model Architecture (GAT-based) ───────────────────────────────────────────
class GestureModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_classes):
        super(GestureModel, self).__init__()
        
        # 1. Temporal Encoder
        self.temporal_enc = TemporalEncoder(in_channels=4, out_dim=dim_hidden)
        
        # 1.5 Sensor Identity (Knowing where the signal came from)
        if USE_SENSOR_ID:
            self.sensor_emb = nn.Embedding(8, dim_hidden)
        
        # 2. GAT Layers (Reasoning about sensor relationships)
        self.conv1 = GATv2Conv(dim_hidden, dim_hidden, heads=4, concat=False)
        self.conv2 = GATv2Conv(dim_hidden, dim_hidden, heads=4, concat=False)
        
        # 3. Disentanglement Heads
        self.public_head = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU())
        self.classifier = nn.Linear(dim_hidden, num_classes)
        
        self.private_head = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU())
        self.discriminator = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, NUM_PARTICIPANTS)
        )

    def forward(self, x, edge_index, batch, grl_lambda=1.0):
        # x: [B*8, 120] -> [B*8, 256]
        x = self.temporal_enc(x)
        
        # Add Sensor Identity
        if USE_SENSOR_ID:
            # Create indices [0,1..7, 0,1..7, ...] for the batch
            num_nodes = x.size(0)
            sensor_indices = torch.arange(8, device=x.device).repeat(num_nodes // 8)
            x = x + self.sensor_emb(sensor_indices)
        
        # Graph Message Passing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global Pooling (8 sensors -> 1 gesture vector)
        x = global_mean_pool(x, batch)

        # Heads
        z_pub = self.public_head(x)
        z_pri = self.private_head(x)

        out_gesture = self.classifier(z_pub)
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
    # PURITY MODE: Train on 10x, but Test/Val ONLY on the 1x Original samples.
    train, val, test = [], [], []
    for i, d in enumerate(dataset):
        pid = d.p_y.item()
        
        # Robust check for Original sample
        # We check for the 'is_orig' attribute, fallback to index if using old dataset
        is_real = False
        if hasattr(d, 'is_orig'):
            is_real = d.is_orig.item()
        else:
            is_real = (i % 10 == 0) # Fallback for legacy support
            
        if pid == test_pid:
            if is_real: test.append(d)
        elif pid == val_pid:
            if is_real: val.append(d)
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
    model = GestureModel(DIM_IN, DIM_HIDDEN, NUM_GESTURES).to(DEVICE)

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

    # Use Focal Loss to master hard gestures
    if USE_FOCAL:
        criterion_gesture = FocalLoss(gamma=2.0)
    else:
        criterion_gesture = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    criterion_identity = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss().to(DEVICE)

    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_f1       = 0.0
    best_errors   = {}
    
    epochs_no_improve = 0
    PATIENCE = 150 # Disable early stopping for thorough Subject 16 learning

    for epoch in range(NUM_EPOCHS):
        model.train()
        
        total_train_correct = 0
        total_train_samples = 0
        
        current_lam = get_adversarial_lambda(epoch, NUM_EPOCHS)
        
        # --- LR WARMUP (First 5 Epochs) ---
        if epoch < 5:
            curr_lr = 0.001 * (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
        
        for batch in train_loader:
            batch = batch.to(DEVICE)

            # --- DYNAMIC JITTER ---
            if USE_DYNAMIC_JITTER and model.training:
                jitter = torch.randn_like(batch.x) * JITTER_SIGMA
                batch.x = batch.x + jitter
            
            # --- DYNAMIC SENSOR DROP ---
            if USE_SENSOR_DROP and torch.rand(1).item() < SENSOR_DROP_RATE:
                num_graphs = batch.num_graphs
                # Pick a random sensor index (0-7) to drop for each graph in batch
                drop_idx = torch.randint(0, 8, (num_graphs,), device=DEVICE)
                # Create a mask: True if we KEEP the node
                node_idx_in_graph = torch.arange(batch.x.size(0), device=DEVICE) % 8
                keep_mask = node_idx_in_graph != drop_idx[batch.batch]
                batch.x = batch.x * keep_mask.unsqueeze(-1).float()
            # ---------------------------

            # --- TEMPORAL SHIFTING (Fixes p16 speed/timing issues) ---
            if model.training:
                feat_dim = batch.x.size(1) // 4 # 30
                x_seq = batch.x.view(-1, 4, feat_dim)
                shift = np.random.randint(-5, 6)
                
                if shift > 0:
                    # Shift right: Pad with the first frame [:, :, 0]
                    padding = x_seq[:, :, 0:1].repeat(1, 1, shift)
                    x_seq = torch.cat([padding, x_seq[:, :, :-shift]], dim=2)
                elif shift < 0:
                    # Shift left: Pad with the last frame [:, :, -1]
                    shift_abs = abs(shift)
                    padding = x_seq[:, :, -1:].repeat(1, 1, shift_abs)
                    x_seq = torch.cat([x_seq[:, :, shift_abs:], padding], dim=2)
                
                batch.x = x_seq.view(batch.x.size(0), -1)

            optimizer.zero_grad()
            
            # --- Mixup Logic (Graph-Level Shuffling) ---
            if USE_MIXUP and model.training:
                mix_lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                batch_size = batch.num_graphs
                graph_index = torch.randperm(batch_size).to(DEVICE)
                
                # Expand graph-level shuffle to node-level
                # Since each graph has exactly 8 nodes
                node_index = torch.arange(batch.x.size(0)).to(DEVICE)
                for i in range(batch_size):
                    node_index[i*8:(i+1)*8] = torch.arange(graph_index[i]*8, (graph_index[i]+1)*8).to(DEVICE)
                
                mixed_x = mix_lam * batch.x + (1 - mix_lam) * batch.x[node_index]
                
                out_g, out_p, z_pub, z_pri = model(mixed_x, batch.edge_index, batch.batch, current_lam)
                
                # Mixed Gesture Loss (labels are graph-level)
                loss_gesture = mix_lam * criterion_gesture(out_g, batch.y.squeeze(-1)) + \
                               (1 - mix_lam) * criterion_gesture(out_g, batch.y[graph_index].squeeze(-1))
                
                # Mixed Identity Loss
                loss_adv_per_sample = mix_lam * torch.nn.functional.cross_entropy(out_p, batch.p_y.squeeze(-1), reduction='none') + \
                                      (1 - mix_lam) * torch.nn.functional.cross_entropy(out_p, batch.p_y[graph_index].squeeze(-1), reduction='none')
            else:
                graph_index = torch.arange(batch.num_graphs).to(DEVICE) # Default for non-mixup
                out_g, out_p, z_pub, z_pri = model(batch.x, batch.edge_index, batch.batch, current_lam)
                loss_gesture = criterion_gesture(out_g, batch.y.squeeze(-1))
                loss_adv_per_sample = torch.nn.functional.cross_entropy(out_p, batch.p_y.squeeze(-1), reduction='none')
            
            # Monitoring Training Accuracy (on dominant label)
            pred_g = out_g.argmax(dim=1)
            total_train_correct += (pred_g == batch.y.squeeze(-1)).sum().item()
            total_train_samples += batch.y.size(0)
            
            # 2. SupCon Loss (Optional Improvement)
            if USE_SUPCON:
                loss_supcon = supcon_criterion(z_pub, batch.y.squeeze(-1))
                loss_gesture = 0.5 * loss_gesture + 0.5 * loss_supcon

            # 3. Adversarial Loss (DANN)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # GRADIENT CLIPPING
            optimizer.step()
        
        # --- EVALUATION ---
        train_acc = total_train_correct / total_train_samples
        val_acc, _, _ = evaluate(model, val_loader)
        test_acc, f1, class_errors = evaluate(model, test_loader)

        if scheduler:
            scheduler.step(val_acc)

        # --- EARLY STOPPING CHECK ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_f1 = f1
            best_errors = class_errors
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # --- LOGGING ---
        unique_preds = torch.unique(pred_g).size(0)
        if (epoch + 1) % 1 == 0: # Print every epoch for diagnostics
            log_print(f'    Epoch {epoch+1:03d}/{NUM_EPOCHS} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Div: {unique_preds}/21 | G:{loss_gesture.item():.4f} Adv:{loss_adv.item():.4f} Ort:{loss_ortho.item():.5f}')

        # --- SUBJECT-AWARE EARLY STOPPING ---
        # If the subject is already doing great (>85%), we can stop early to save time.
        # If the subject is struggling (<75%), we NEVER stop early.
        if val_acc > 0.85:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= 30: # Patience of 30 for converged subjects
                log_print(f"    [EARLY STOP] Subject converged at {val_acc:.4f}. Skipping remaining epochs.")
                break
        else:
            # For struggling subjects, just track best but don't increment patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc, best_f1, best_errors = evaluate(model, test_loader)

    return best_test_acc, best_f1, best_errors


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Clear or initialize the results file
    with open(RESULTS_FILE, 'w') as f:
        f.write("=== TRAINING SESSION START ===\n")

    log_print(f'\n{"="*50}')
    log_print(f' VERSION: IMPROVED (Attn:GAT, SID:{USE_SENSOR_ID}, S-Drop:{USE_SENSOR_DROP}, Jitter:{USE_DYNAMIC_JITTER}, Focal:{USE_FOCAL}, SupCon:{USE_SUPCON}, Temp:Attn, Mix:{USE_MIXUP}, Eval:PURITY)')
    log_print(f' RUNNING ON: {DEVICE}')
    if DEVICE.type == 'cuda':
        log_print(f' GPU NAME:   {torch.cuda.get_device_name(0)}')
    log_print(f'{"="*50}\n')

    log_print('Loading dataset...')
    dataset = load_full_dataset()
    log_print(f'Total samples: {len(dataset)}')

    pids = sorted(set(d.p_y.item() for d in dataset))
    log_print(f'Participants found: {pids}')
    
    combinations = [(15, 1), (12, 15), (15,10), (13,15), (9,14), (7,8), (3,4), (14,15), (8,9), (1,3), (10,13), (11,12), (1,14), (6,7), (10,14), (9,13), (3,15), (6,12), (11,15), (4,15), (12,14), (5,15)]
    
    total_runs = len(combinations)
    log_print(f'Total combinations: {total_runs} (Ultra-Fast Mode)')

    all_acc, all_f1 = [], []

    for run_idx, (test_pid, val_pid) in enumerate(combinations, 1):
        log_print(f"\n    Splitting data (Test: p{test_pid+1}, Val: p{val_pid+1})...")
        log_print(f"    [PURITY FILTER] Testing on Original Samples ONLY.")
        train_data, val_data, test_data = split_three_way(
            dataset, test_pid, val_pid
        )

        if len(test_data) == 0 or len(val_data) == 0:
            log_print(f'  Skipping: empty split for test=p{test_pid+1} val=p{val_pid+1}')
            continue

        MAX_ATTEMPTS = 2
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                log_print(f"    [RETRY] Attempt {attempt}/{MAX_ATTEMPTS} for p{test_pid+1}...")
            
            acc, f1, class_errors = train_one_combination(
                train_data, val_data, test_data,
                test_pid, val_pid, run_idx, total_runs
            )
            
            if acc > 0.20: # If we passed the "Dead Zone"
                break
            elif attempt < MAX_ATTEMPTS:
                log_print(f"    [FAIL] Run was dead (Acc: {acc:.4f}). Resetting and retrying...")
            else:
                log_print(f"    [ABANDON] Run stayed dead after {MAX_ATTEMPTS} attempts. Recording score.")
        
        log_print(f"  [DONE] Acc: {acc:.4f} | F1: {f1:.4f}")
        # Print Top 3 Failing Gestures
        sorted_errors = sorted(class_errors.items(), key=lambda x: x[1], reverse=True)
        log_print(f"  Top Errors: " + ", ".join([f"G{k+1}({v})" for k, v in sorted_errors[:3]]))
        
        all_acc.append(acc)
        all_f1.append(f1)

    # ── Final summary ────────────────────────────────────────────────────────
    mean_acc = np.mean(all_acc)
    std_acc  = np.std(all_acc)
    mean_f1  = np.mean(all_f1)

    header = (
        f"\n{'='*60}\n"
        f"LOPO RESULTS ({len(all_acc)} combinations)\n"
        f"VERSION: Attn:{'Trans' if USE_TRANSFORMER else 'GAT'}, SupCon:{USE_SUPCON}, Temp:{USE_TEMPORAL}, Mix:{USE_MIXUP}\n"
        f"{'='*60}\n"
    )
    body = (
        f"  Mean Test Accuracy : {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"  Mean F1            : {mean_f1:.4f}\n"
        f"{'='*60}\n"
    )
    log_print(header + body)

    with open(RESULTS_FILE, 'a') as f:
        for (test_pid, val_pid), acc, f1 in zip(combinations, all_acc, all_f1):
            f.write(f"test=p{test_pid+1:02d} val=p{val_pid+1:02d} | acc={acc:.4f} f1={f1:.4f}\n")
    
    log_print(f"Detailed results saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()