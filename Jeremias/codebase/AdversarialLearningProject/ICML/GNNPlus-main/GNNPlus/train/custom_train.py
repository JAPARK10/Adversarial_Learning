import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv

# Load toggles from .env
load_dotenv()
USE_ADVERSARIAL = os.getenv("USE_ADVERSARIAL_LAYERS", "false").lower() == "true"
USE_CONTRASTIVE = os.getenv("USE_CONTRASTIVE_LEARNING", "false").lower() == "true"

from GNNPlus.loss.subtoken_prediction_loss import subtoken_cross_entropy
from GNNPlus.loss.supcon_loss import SupConLoss

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from GNNPlus.utils import cfg_to_dict, flatten_dict, make_wandb_name, dirichlet_energy, mean_average_distance, mean_norm

def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation, cur_epoch):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        extra_stats = {}
        
        # New multi-head output
        preds_dict, true = model(batch)
        
        # Main Exercise Loss
        ex_pred = preds_dict['exercise']
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(ex_pred, true)
        else:
            loss, pred_score = compute_loss(ex_pred, true)
            
        # Optional Adversarial Loss
        if USE_ADVERSARIAL and 'participant' in preds_dict:
            p_pred = preds_dict['participant']
            p_true = batch.p_y
            p_loss = F.cross_entropy(p_pred, p_true.view(-1))
            
            # --- DANN: LOGISTIC WARMUP ---
            # A smooth transition across training epochs prevents 'Gradient Shock'.
            # p goes from 0 to 1 over the course of training.
            max_epochs = cfg.optim.max_epoch
            p = float(cur_epoch) / max_epochs
            # standard DANN formula: lambda = 2 / (1 + exp(-10 * p)) - 1
            lambda_adv = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            
            # Cap the weight at a safe threshold to prevent over-regularization
            lambda_adv = min(lambda_adv, 1.0)
            
            loss = loss + lambda_adv * p_loss 
            
            # Monitor unlearning progress
            p_acc = (p_pred.argmax(dim=1) == p_true.view(-1)).float().mean().item()
            extra_stats['p_acc'] = p_acc
            extra_stats['lambda_adv'] = lambda_adv
            
        # Optional Contrastive Loss (SupCon)
        if USE_CONTRASTIVE and 'contrastive' in preds_dict:
            con_emb = preds_dict['contrastive']
            criterion_con = SupConLoss(temperature=0.07)
            con_loss = criterion_con(con_emb, true)
            loss = loss + 0.1 * con_loss # Lambda_Con = 0.1
            
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
            
        if cfg.train.eval_smoothing_metrics:
            extra_stats['dirichlet'] = dirichlet_energy(batch.x, batch.edge_index, batch.batch)
            extra_stats['mad'] = mean_average_distance(batch.x, batch.edge_index, batch.batch)
            extra_stats['emb_norm'] = mean_norm(batch.x)
            
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        
        time_start = time.time()




#######WithTestAccuracyOnly########
# @torch.no_grad()
# def eval_epoch(logger, loader, model, split='val'):
#     model.eval()
#     time_start = time.time()
#     for batch in loader:
#         batch.split = split
#         batch.to(torch.device(cfg.accelerator))
#         if cfg.gnn.head == 'inductive_edge':
#             pred, true, extra_stats = model(batch)
#         else:
#             pred, true = model(batch)
#             extra_stats = {}
#         if cfg.dataset.name == 'ogbg-code2':
#             loss, pred_score = subtoken_cross_entropy(pred, true)
#             _true = true
#             _pred = pred_score
#         else:
#             loss, pred_score = compute_loss(pred, true)
#             _true = true.detach().to('cpu', non_blocking=True)
#             _pred = pred_score.detach().to('cpu', non_blocking=True)
        
#         if cfg.train.eval_smoothing_metrics:
#             extra_stats['dirichlet'] = dirichlet_energy(batch.x, batch.edge_index, batch.batch)
#             extra_stats['mad'] = mean_average_distance(batch.x, batch.edge_index, batch.batch)
#             extra_stats['emb_norm'] = mean_norm(batch.x)
            
#         logger.update_stats(true=_true,
#                             pred=_pred,
#                             loss=loss.detach().cpu().item(),
#                             lr=0, time_used=time.time() - time_start,
#                             params=cfg.params,
#                             dataset_name=cfg.dataset.name,
#                             **extra_stats)
#         time_start = time.time()

#######TestAccuracyWithPreRecF1Sco########
@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    
    # --- ADD THESE LISTS TO COLLECT ALL DATA ---
    all_true = []
    all_pred = []
    # --------------------------------------------

    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            preds_dict, true = model(batch)
            pred = preds_dict['exercise']
            extra_stats = {}
        
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)

        # --- COLLECT DATA FOR EVALUATION ---
        all_true.append(_true)
        all_pred.append(_pred)
        # -----------------------------------

        if cfg.train.eval_smoothing_metrics:
            extra_stats['dirichlet'] = dirichlet_energy(batch.x, batch.edge_index, batch.batch)
            extra_stats['mad'] = mean_average_distance(batch.x, batch.edge_index, batch.batch)
            extra_stats['emb_norm'] = mean_norm(batch.x)
            
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()

    # --- CALCULATE PRECISION, RECALL, F1 AFTER THE LOOP ---
    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    
    # Convert logits to class labels
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # We return these so custom_train can access them directly
    metrics = {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)
        
    # --- RIGOROUS SIZE VERIFICATION ---
    logging.info("\n" + "-"*40)
    logging.info("      [DATASET SIZE VERIFICATION]")
    logging.info("-"*40)
    split_names = ['Train', 'Val', 'Test']
    for i, loader in enumerate(loaders):
        logging.info(f"[*] {split_names[i]:<6} samples: {len(loader.dataset)}")
    logging.info("-"*40 + "\n")

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation, cur_epoch)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                # CAPTURE THE RETURNED METRICS HERE
                extra_metrics = eval_epoch(loggers[i], loaders[i], model,
                                           split=split_names[i - 1])
                
                # GET THE STANDARD STATS FROM THE LOGGER
                epoch_stats = loggers[i].write_epoch(cur_epoch)
                
                # MERGE THEM
                epoch_stats.update(extra_metrics)
                perf[i].append(epoch_stats)
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                        for x in ['dirichlet', 'mad', 'emb_norm']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            
            if best_epoch == cur_epoch:                    
                # Create directory if it doesn't exist
                conf_dir = os.path.join(cfg.run_dir, "Conf")
                if not os.path.exists(conf_dir):
                    os.makedirs(conf_dir)

                model.eval()
                y_true_list, y_pred_list = [], []
                for cm_batch in loaders[2]: # Test loader
                    cm_batch.to(torch.device(cfg.accelerator))
                    with torch.no_grad():
                        cm_preds_dict, cm_true = model(cm_batch)
                    cm_pred = cm_preds_dict['exercise']
                    y_true_list.append(cm_true.cpu().numpy())
                    y_pred_list.append(cm_pred.argmax(dim=1).cpu().numpy())
                    
                cm_true_all = np.concatenate(y_true_list)
                cm_pred_all = np.concatenate(y_pred_list)
                conf_mat = confusion_matrix(cm_true_all, cm_pred_all, normalize='true')
                annot_labels = np.full(conf_mat.shape, "", dtype=object)


                for i in range(conf_mat.shape[0]):
                    for j in range(conf_mat.shape[1]):
                        if conf_mat[i, j] > 0:
                            annot_labels[i, j] = f"{conf_mat[i, j]:.2f}"
                    
                plt.figure(figsize=(10, 8))
                sns.heatmap(conf_mat, annot=annot_labels, fmt="", cmap='Blues', vmin=0, vmax=1)
                # plt.title(f'Confusion Matrix (Test Set) - Epoch {cur_epoch}')
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                    
                file_name = f'confusion_matrix_epoch_{cur_epoch}.png'
                cm_save_path = os.path.join(conf_dir, file_name)
                plt.savefig(cm_save_path)
                plt.close()
                logging.info(f"[*] Confusion Matrix saved to: {cm_save_path}")
            
            # Inside the is_eval_epoch(cur_epoch) block
            test_f1 = perf[2][best_epoch].get('f1', 0)
            test_pre = perf[2][best_epoch].get('precision', 0)
            test_rec = perf[2][best_epoch].get('recall', 0)
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}\t"
                f"F1: {test_f1:.4f} Prec: {test_pre:.4f} Rec: {test_rec:.4f}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    with open(f'results/{cfg.dataset.name}_result.txt','a') as f:
        f.write(f'{cfg.gnn.layer_type} '+f'residual_{cfg.gnn.residual} '+f'ffn_{cfg.gnn.ffn} '+f'{cfg.gnn.layers_mp} '+f'{cfg.gnn.dim_inner} '+f'{cfg.gnn.dropout} seed_{cfg.seed}: ')
        f.write(f'{best_test}\n')
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)
    return perf[2][best_epoch] # Return best test stats


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


@register_train('PCQM4Mv2-inference')
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ['valid', 'test-dev', 'test-challenge']
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert(all([not torch.isnan(d.y)[0] for d in loaders[0].dataset]))
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[1].dataset]))
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[2].dataset]))

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            all_true.append(true.detach().to('cpu', non_blocking=True))
            all_pred.append(pred.detach().to('cpu', non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {'y_pred': all_pred.squeeze(),
                          'y_true': all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")  # Get MAE.
        else:
            input_dict = {'y_pred': all_pred.squeeze()}
            evaluator.save_test_submission(input_dict=input_dict,
                                           dir_path=cfg.run_dir,
                                           mode=split_names[i])


@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from GNNPlus.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.accelerator))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
