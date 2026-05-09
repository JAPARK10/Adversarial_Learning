from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('adv')
def set_cfg_adv(cfg):
    """
    Adversarial parameters for Gradient Reversal Layer based user classification.
    """
    cfg.adv = CN()
    cfg.adv.use = False
    cfg.adv.lambda_u = 0.1
    cfg.adv.num_users = 16


@register_config('windowed')
def set_cfg_windowed(cfg):
    """
    Windowed graph construction flag.
    When True, each gesture (30, 8, 2) is split into 3 non-overlapping windows
    of 10 time steps, processed by the shared GNN backbone, and combined via
    concatenation before the gesture head.
    """
    cfg.dataset.windowed = False  # Default: single-graph baseline
