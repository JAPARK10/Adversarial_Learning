#temporarily

import sys
sys.argv = [
    "main.py",
    "--cfg", "configs/gcn/rfid.yaml",
    "--repeat", "1",
    "seed", "0"
]


import datetime
import torch
import warnings

# Suppress PyG's noisy InMemoryDataset warnings
warnings.filterwarnings("ignore", message=".*InMemoryDataset.*")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

try:
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr
    # Allow PyG graph objects in PyTorch 2.6+ (which defaults to weights_only=True)
    torch.serialization.add_safe_globals([Data, DataEdgeAttr])
except ImportError:
    pass
import logging
import os
from dotenv import load_dotenv

# Load toggles from .env
load_dotenv()
USE_ADVERSARIAL = os.getenv("USE_ADVERSARIAL_LAYERS", "false").lower() == "true"
USE_CONTRASTIVE = os.getenv("USE_CONTRASTIVE_LEARNING", "false").lower() == "true"
USE_PERSON_EXCLUSIVE = os.getenv("USE_PERSON_EXCLUSIVE_SPLIT", "false").lower() == "true"
EXCLUDE_VAL_ID = os.getenv("EXCLUDE_PERSON_ID_VAL", "15")
EXCLUDE_TEST_ID = os.getenv("EXCLUDE_PERSON_ID_TEST", "16")
USE_FULL_LOPO_CYCLE = os.getenv("USE_FULL_LOPO_CYCLE", "false").lower() == "true"
USE_STRESS_TEST = os.getenv("USE_STRESS_TEST", "false").lower() == "true"

import GNNPlus  # noqa, register custom modules
from GNNPlus.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, set_cfg, load_cfg)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from GNNPlus.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from GNNPlus.logger import create_logger


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg."""
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


def final_comprehensive_report(model, loader):
    """
    Runs a final pass over the test loader and reports accuracy per participant ID.
    """
    model.eval()
    p_correct = {}
    p_total = {}
    
    for batch in loader:
        batch.to(torch.device(cfg.accelerator))
        with torch.no_grad():
            preds_dict, targets = model(batch)
        preds = preds_dict['exercise'].argmax(dim=1)
        
        # batch.p_y contains participant IDs
        p_ids = batch.p_y.view(-1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        for i in range(len(p_ids)):
            p = int(p_ids[i])
            is_correct = 1 if preds_np[i] == targets_np[i] else 0
            p_correct[p] = p_correct.get(p, 0) + is_correct
            p_total[p] = p_total.get(p, 0) + 1
            
    # Print the table
    logging.info("\n" + "="*50)
    logging.info("      [PER-PARTICIPANT GENERALIZATION REPORT]")
    logging.info("="*50)
    logging.info(f"{'Subject ID':<12} | {'Accuracy':<10} | {'Samples':<8}")
    logging.info("-" * 40)
    
    total_samples = 0
    weighted_acc = 0
    for p in sorted(p_correct.keys()):
        acc = p_correct[p] / p_total[p]
        logging.info(f"Subject #{p:<5} | {acc*100:>8.2f}% | {p_total[p]:<8}")
        total_samples += p_total[p]
        weighted_acc += p_correct[p]
        
    if total_samples > 0:
        final_avg = weighted_acc / total_samples
        logging.info("-" * 40)
        logging.info(f"{'OVERALL AGG':<12} | {final_avg*100:>8.2f}% | {total_samples:<8}")
    logging.info("="*50 + "\n")


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()

    # Determine combinations to run
    if USE_FULL_LOPO_CYCLE:
        # Override individual IDs and run all 16 subjects
        num_participants = 16
        combinations = [(i, (i + 1) % num_participants) for i in range(num_participants)]
        logging.info(f"[*] LOPO MODE: Full 16-run cycle enabled.")
    else:
        # Run the single pair defined in .env
        combinations = [(int(EXCLUDE_TEST_ID), int(EXCLUDE_VAL_ID))]
        logging.info(f"[*] LOPO MODE: Single pair (T:{EXCLUDE_TEST_ID}, V:{EXCLUDE_VAL_ID})")

    # Start the execution loop
    for test_id, val_id in combinations:
        # Set configurations for this specific LOPO pair
        set_cfg(cfg)
        load_cfg(cfg, args)
        
        # Inject IDs into environment so split_generator can see them
        os.environ["EXCLUDE_PERSON_ID_TEST"] = str(test_id)
        os.environ["EXCLUDE_PERSON_ID_VAL"] = str(val_id)
        
        # Dynamic Naming for Run Directory
        adv_str = "AdvT" if USE_ADVERSARIAL else "AdvF"
        con_str = "ConT" if USE_CONTRASTIVE else "ConF"
        
        if USE_STRESS_TEST:
            pel_str = "STRESS_TEST"
        else:
            pel_str = f"PX_V{val_id}_T{test_id}" if USE_PERSON_EXCLUSIVE else "PXF"
            
        tag = f"{adv_str}_{con_str}_{pel_str}_OPT"
        
        if "results" in cfg.run_dir:
            cfg.run_dir = cfg.run_dir.replace("results", f"results_{tag}")
        else:
            cfg.run_dir = f"{cfg.run_dir}_{tag}"
            
        logging.info(f"\n{'='*60}")
        logging.info(f"[*] STARTING RUN: Test=p{test_id:02d}, Val=p{val_id:02d}")
        logging.info(f"[*] Run Directory: {cfg.run_dir}")
        logging.info(f"{'='*60}\n")

        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)

        # Inner GraphGym repeat loop (usually just 1)
        for run_id, seed, split_index in zip(*run_loop_settings()):
            set_printing()
            cfg.dataset.split_index = split_index
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            auto_select_device()
            
            # Load machine learning pipeline
            # Note: create_loader calls split_generator.prepare_splits internally
            loaders = create_loader()
            loggers = create_logger()
            model = create_model()
            
            model.to(torch.device(cfg.accelerator))
            optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            
            # Start training
            best_stats = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)
        
            # Reload the best model checkpoint before generating the final test report
            if cfg.train.enable_ckpt and cfg.train.ckpt_best:
                try:
                    import glob
                    ckpt_dir = os.path.join(cfg.run_dir, 'ckpt')
                    ckpts = glob.glob(f"{ckpt_dir}/*.ckpt")
                    if ckpts:
                        latest_ckpt = max(ckpts, key=os.path.getctime)
                        ckpt = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
                        model.load_state_dict(ckpt['model_state'])
                        logging.info(f"[*] Reloaded best checkpoint for final report.")
                except Exception as e:
                    logging.warning(f"[W] Failed to reload best checkpoint: {e}")
            
            # This generates the detailed accuracy table for the test subject
            final_comprehensive_report(model, loaders[2])

    logging.info(f"[*] All runs completed: {datetime.datetime.now()}")
