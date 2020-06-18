from torch import nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.reduce:
            CE_Loss = F.cross_entropy(inputs, targets, reduction='mean')
        else:
            CE_Loss = F.cross_entropy(inputs, targets, reduction=None)
        pt = torch.exp(-CE_Loss)
        F_Loss = self.alpha * (1-pt)**self.gamma * CE_Loss

        return F_Loss