from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config

@register_config('adv')
def set_cfg_adv(cfg):
    """
    Configuration for E-UIGR Adversarial framework.
    """
    cfg.adv = CN()
    
    # Enable adversarial learning
    cfg.adv.use = False
    
    # Lambda weight for the Gradient Reversal Layer
    cfg.adv.lambda_u = 3.5
    
    # Number of users (participants) in the dataset to size the adversarial head
    cfg.adv.num_users = 16
