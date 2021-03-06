from torch import nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):

        if self.reduce:
            BCE_Loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='mean')
        else:
            BCE_Loss = F.binary_cross_entropy(inputs, targets.float(), reduction=None)

        pt = torch.exp(-BCE_Loss)

        F_Loss = self.alpha * (1-pt)**self.gamma * BCE_Loss

        return F_Loss