from torch import nn
from torch.nn import functional as F


def calculate_dice(inputs, targets, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice


class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()


class DiceCoeffecient(Dice):
    def __init__(self, weight=None, size_average=True):
        super(DiceCoeffecient, self).__init__(weight, size_average)

    def forward(self, inputs, targets, smooth=1):
        return calculate_dice(inputs, targets, smooth)


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def calculate_iou(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        return (intersection + smooth) / (union + smooth)


class IoUCoeffecient(IoU):
    def __init__(self, weight=None, size_average=True):
        super(IoUCoeffecient, self).__init__(weight, size_average)

    def forward(self, inputs, targets, smooth=1):
        return self.calculate_iou(inputs, targets, smooth)


class DiceBCELoss(Dice):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__(weight, size_average)

    def forward(self, inputs, targets, smooth=1):
        dice_loss = 1 - calculate_dice(inputs, targets, smooth)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
