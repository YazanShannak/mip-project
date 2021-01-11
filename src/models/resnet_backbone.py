import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from .utils import DiceCoefficient, DiceBCELoss, IoUCoefficient


class ResNetBackBone(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, gamma: float = 0.5):
        super(ResNetBackBone, self).__init__()
        self.model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1,
                              activation="sigmoid")
        self.lr = lr
        self.gamma = gamma
        self.criterion = DiceBCELoss()
        self.dice = DiceCoefficient()
        self.iou = IoUCoefficient()

    def forward(self, x):
        output = self.model(x)
        return output

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
        iou = self.iou(output, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        positive_indices = torch.where(target > 0)[0]
        if len(positive_indices) > 0:
            random_index = torch.randint(low=0, high=len(positive_indices) - 1, size=(1,)).item()
            index = positive_indices[random_index].item()
            self.logger.experiment.add_image("Predicted Mask", torch.where(output[index] > 0.5, 1, 0), self.global_step,
                                             dataformats="CHW")
            self.logger.experiment.add_image("Actual Mask", target[index], self.global_step,
                                             dataformats="CHW")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
