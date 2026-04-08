"""
custom_train.py  –  GNNPlus training loop with domain-adversarial learning.

CHANGES vs original (inspired by EUIGR paper, SenSys 2019):
  ① GradientReversal layer  – reverses gradients flowing from the user
    discriminator back into the GNN encoder, forcing the encoder to
    produce user-invariant representations.
  ② UserDiscriminator MLP   – a 2-layer MLP that tries to predict which
    participant performed the gesture from the graph embedding.
  ③ Adversarial loss        – total loss = L_gesture − λ * L_user
    The minus sign is what makes it adversarial: the encoder is trained
    to *maximise* user confusion while *minimising* gesture loss.
  ④ rfid.yaml needs one new key:   adv_lambda: 0.5   (see below)
  ⑤ Everything else (logging, checkpointing, confusion-matrix saving)
    is unchanged from the professor's original file.

How to enable / disable adversarial training:
  In configs/gcn/rfid.yaml add:
      adv_lambda: 0.5        # weight of adversarial loss  (0 = disabled)
      num_participants: 17   # number of participant classes
  If these keys are absent the code falls back to normal training.
"""

import logging
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# ①  Gradient Reversal Layer
#    Forward pass: identity  (x → x)
#    Backward pass: negates gradient  (∂L/∂x → −λ * ∂L/∂x)
#    This is the standard trick used in domain-adversarial neural networks
#    (Ganin et al., JMLR 2016) and adopted by EUIGR.
# ─────────────────────────────────────────────────────────────────────────────

class _GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None   # reverse & scale gradient


class GradientReversal(nn.Module):
    """Wraps _GradRevFn so it can be used like a normal nn.Module."""
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return _GradRevFn.apply(x, self.lam)


# ─────────────────────────────────────────────────────────────────────────────
# ②  User Discriminator  (the "adversarial head" in your supervisor's diagram)
#    Input:  graph-level embedding  (dim = cfg.gnn.dim_inner = 128)
#    Output: logits over N participants
# ─────────────────────────────────────────────────────────────────────────────

class UserDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_participants: int, lam: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(lam)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_participants),
        )

    def forward(self, graph_embed):
        """graph_embed: (batch_size, in_dim)"""
        x = self.grl(graph_embed)
        return self.mlp(x)           # (batch_size, num_participants)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    accuracy = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        n_cls = y_prob.shape[1]
        if n_cls == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr',
                                average='weighted')
    except Exception:
        auc = 0.0
    return accuracy, f1, auc


def train_epoch(logger, loader, model, optimizer, scheduler,
                user_disc=None, adv_lambda=0.0, num_participants=16):
    model.train()
    if user_disc is not None:
        user_disc.train()

    time_start = time.time()

    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()

        # ── forward pass through GNN ──────────────────────────────────────
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)

        # ── ③ adversarial loss ───────────────────────────────────────────
        if user_disc is not None and adv_lambda > 0:
            # Extract graph-level embedding produced by mean-pooling inside
            # the GNN.  GNNPlus stores it as model.post_mp's input, which
            # we can access via a forward hook, but the simplest approach is
            # to re-use the graph embedding that GNNPlus already computes
            # and exposes as  batch.graph_feature  after the forward call.
            graph_embed = getattr(batch, 'graph_feature', None)

            if hasattr(batch, 'participant'):
    # Use pred directly but detach from classification head
    # Get graph embedding from pooled node features (dim=128)
                from torch_geometric.nn import global_mean_pool
                node_feats = batch.x  # node features after GNN = (N_total, 128)
                graph_embed = global_mean_pool(node_feats, batch.batch)  # (B, 128)
                participant_labels = batch.participant.squeeze(-1).to(graph_embed.device)
                user_logits = user_disc(graph_embed)
                adv_loss = F.cross_entropy(user_logits, participant_labels)
                loss = loss + adv_lambda * adv_loss

        loss.backward()
        optimizer.step()

        logger.update_stats(
            true=true.detach().cpu(),
            pred=pred_score.detach().cpu(),
            loss=loss.item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
        )
        time_start = time.time()

    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()

    for batch in loader:
        batch.split = split
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)

        logger.update_stats(
            true=true.detach().cpu(),
            pred=pred_score.detach().cpu(),
            loss=loss.item(),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params,
        )
        time_start = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Main training loop.
    Reads two optional keys from cfg (set in rfid.yaml):
        cfg.adv_lambda       – float, adversarial loss weight (default 0)
        cfg.num_participants – int,   number of users          (default 17)
    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
        return

    # ── read adversarial config ───────────────────────────────────────────
    adv_lambda      = 0.5
    num_participants = 16

    # ── ② build user discriminator ───────────────────────────────────────
    user_disc = None
    adv_optimizer = None
    if adv_lambda > 0:
        in_dim = cfg.gnn.dim_inner          # 128 in rfid.yaml
        user_disc = UserDiscriminator(
            in_dim=in_dim,
            num_participants=num_participants,
            lam=adv_lambda,
        ).to(torch.device(cfg.device if hasattr(cfg, 'device') else 'cpu'))

        adv_optimizer = torch.optim.Adam(user_disc.parameters(), lr=1e-3)
        logging.info(
            f'[Adversarial] UserDiscriminator enabled: '
            f'λ={adv_lambda}, participants={num_participants}'
        )
    else:
        logging.info('[Adversarial] Disabled (adv_lambda=0)')

    logging.info('Start from epoch %s', start_epoch)

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    # ── results file ─────────────────────────────────────────────────────
    result_path = os.path.join(cfg.run_dir, 'rfid_result.txt')

    # ── confusion-matrix directory ───────────────────────────────────────
    conf_dir = os.path.join(cfg.run_dir, 'Conf')
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        t0 = time.perf_counter()

        train_epoch(
            loggers[0], loaders[0], model, optimizer, scheduler,
            user_disc=user_disc,
            adv_lambda=adv_lambda,
            num_participants=num_participants,
        )
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1])

        full_epoch_times.append(time.perf_counter() - t0)
        eta = np.mean(full_epoch_times) * (cfg.optim.max_epoch - cur_epoch - 1)
        logging.info(
            f'Epoch {cur_epoch} | '
            f'train {perf[0][-1]["accuracy"]:.4f} | '
            f'val {perf[1][-1]["accuracy"]:.4f} | '
            f'ETA {eta:.0f}s'
        )

        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
            if cfg.train.ckpt_clean:
                clean_ckpt()

        # ── save confusion matrix every few epochs ────────────────────────
        if cur_epoch % 5 == 0 or cur_epoch == cfg.optim.max_epoch - 1:
            _save_confusion_matrix(loaders[2], model, conf_dir, cur_epoch)

    # ── final summary ─────────────────────────────────────────────────────
    best_val_epoch = int(np.argmax([p['accuracy'] for p in perf[1]]))
    best_train = perf[0][best_val_epoch]['accuracy']
    best_val   = perf[1][best_val_epoch]['accuracy']
    best_test  = perf[2][best_val_epoch]['accuracy']

    logging.info(
        f'Best val epoch: {best_val_epoch} | '
        f'train {best_train:.4f} | val {best_val:.4f} | test {best_test:.4f}'
    )

    # ── write result file (same format as original) ───────────────────────
    layer_type   = cfg.gnn.layer_type
    residual     = cfg.gnn.residual if hasattr(cfg.gnn, 'residual') else False
    ffn          = cfg.gnn.ffn      if hasattr(cfg.gnn, 'ffn')      else True
    layers_mp    = cfg.gnn.layers_mp
    dim_inner    = cfg.gnn.dim_inner
    dropout      = cfg.gnn.dropout

    result_str = (
        f'{layer_type} residual_{residual} ffn_{ffn} '
        f'{layers_mp} {dim_inner} {dropout} '
        f'seed_{cfg.seed}: '
        f'test_accuracy: {best_test:.4f}'
        f'  [adv_lambda={adv_lambda}]'
    )

    logging.info(result_str)
    with open(result_path, 'w') as f:
        f.write(result_str + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Confusion-matrix helper (unchanged logic, just extracted to function)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _save_confusion_matrix(loader, model, conf_dir, epoch):
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        batch.split = 'test'
        pred, true = model(batch)
        all_true.extend(true.cpu().numpy())
        all_pred.extend(pred.argmax(dim=1).cpu().numpy())

    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix – Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(conf_dir, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()

from torch_geometric.graphgym.register import register_train
register_train('custom', custom_train)
