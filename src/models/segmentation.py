import torch
import pytorch_lightning as pl
from src.models.unet import UnetEncoder, UnetDecoder
from src.models.utils import DiceCoefficient, IoUCoefficient, DiceBCELoss


class SegmentationUnet(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, gamma: float = 0.5, freeze_encoder: bool = False):
        super(SegmentationUnet, self).__init__()
        self.save_hyperparameters()
        self.encoder = UnetEncoder()
        self.decoder = UnetDecoder()

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.lr = lr
        self.gamma = gamma
        self.criterion = DiceBCELoss()
        self.dice = DiceCoefficient()
        self.iou = IoUCoefficient()

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        output = self.decoder(*encoder_outputs)
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
            self.logger.experiment.add_image("Predicted Mask", output[index], self.global_step,
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
