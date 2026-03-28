import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        """
        device = features.device
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        # For numerical stability, use logsumexp pattern
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero for samples that have no other same-class partners in the batch
        mask_pos_pairs = torch.where(mask_pos_pairs > 0, mask_pos_pairs, torch.ones_like(mask_pos_pairs))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = - mean_log_prob_pos
        # Only average over samples that actually HAD positive pairs
        valid_indices = (mask.sum(1) > 0)
        if valid_indices.any():
            loss = loss[valid_indices].mean()
        else:
            loss = loss.mean() * 0.0 # Zero loss if no pairs found

        return loss
