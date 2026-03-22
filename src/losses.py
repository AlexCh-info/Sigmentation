import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    diceloss - хорошо работает при дисбалансе классов
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)


        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice

class BCEDiceLoss(nn.Module):
    """
    Binary crossentropy + DiceLoss
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        loss_dice = self.dice(pred_sigmoid, target)

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice