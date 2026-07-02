import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss implementation to address class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: tensor of arbitrary shape (usually batch_size or batch_size x seq_len)
        targets: tensor of the same shape containing 0 or 1 labels
        """
        # Ensure targets are float
        targets = targets.float()
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Clamp to avoid numerical instability of log(0)
        eps = 1e-8
        probs = torch.clamp(probs, min=eps, max=1.0 - eps)
        
        # Standard Binary Cross Entropy
        bce = -targets * torch.log(probs) - (1.0 - targets) * torch.log(1.0 - probs)
        
        # Focal weight (1 - p_t)^gamma
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        
        loss = focal_weight * bce
        
        # Balance parameter alpha
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
