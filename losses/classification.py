from __future__ import annotations
import torch
import torch.nn.functional as F


def cross_entropy_with_label_smoothing(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing <= 0:
        return F.cross_entropy(logits, targets)

    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(label_smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(true_dist * log_probs).sum(dim=-1).mean()
