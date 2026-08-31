"""Loss helpers.

Reconstructed to satisfy `from utils.loss import *` in the ResNet entry point.
The DGMSNet ComposerModel computes its own `F.cross_entropy` loss, so nothing
here is strictly required at runtime; these are provided for completeness and
so the wildcard import resolves.
"""
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CrossEntropyLoss", "cross_entropy"]


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    def forward(self, logits, targets):
        return self.criterion(logits, targets)


def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, **kwargs)
