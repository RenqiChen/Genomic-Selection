import os
import numpy as np

import torch
import torch.nn as nn

def build_loss(args):
    loss_name = args.loss_name
    if loss_name == "mse_loss":
        loss = mse_loss(args)
    if loss_name == "ce_loss":
        loss = ce_loss(args)
    return loss


class mse_loss(nn.Module):
    def __init__(self, args):
        super(mse_loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=args.reduction)

    def forward(self, predictions, targets, mask=None):
        predictions = predictions.squeeze(-1)
        targets = targets.squeeze(-1)
        assert predictions.shape == targets.shape
        if mask is not None:
            loss = self.mse_loss(predictions * mask, targets * mask)
        else:
            loss = self.mse_loss(predictions, targets)
        return loss
    

class ce_loss(nn.Module):
    def __init__(self, args):
        super(ce_loss, self).__init__()
        # weights = torch.tensor(args.class_weights, device=args.device)
        # self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, mask=None):
        loss = self.ce_loss(predictions, targets.to(torch.long))
        return loss