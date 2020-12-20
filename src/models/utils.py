from torch import nn
from torch.nn import functional as F


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    @staticmethod
    def calculate_dice(inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice

    def forward(self, inputs, targets, smooth=1):
        pass


class DiceCoefficient(Dice):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        return self.calculate_dice(inputs, targets, smooth)


class DiceBCELoss(Dice):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        dice_loss = 1 - self.calculate_dice(inputs, targets, smooth)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    @staticmethod
    def calculate_iou(inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        return (intersection + smooth) / (union + smooth)


class IoUCoefficient(IoU):
    def __init__(self):
        super(IoUCoefficient, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        return self.calculate_iou(inputs, targets, smooth)
