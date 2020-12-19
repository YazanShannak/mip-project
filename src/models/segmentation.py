import torch
import pytorch_lightning as pl
from src.models.unet import Unet
from src.models.utils import DiceCoeffecient, IoUCoeffecient, DiceBCELoss


class SegmentationUnet(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super(SegmentationUnet, self).__init__()
        self.save_hyperparameters()
        self.model = Unet()

        self.lr = lr
        self.criterion = DiceBCELoss()
        self.dice = DiceCoeffecient()
        self.iou = IoUCoeffecient()

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.criterion(output, target)

        dice = self.dice(output, target)
        iou = self.iou(output, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice", dice, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_iou", iou, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.criterion(output, target)

        dice = self.dice(output, target)
        iou = self.dice(output, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer
