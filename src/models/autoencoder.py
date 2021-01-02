import torch
import pytorch_lightning as pl
from src.models.unet import Unet
from torch import nn


class AutoEncoder(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super(AutoEncoder, self).__init__()
        self.model = Unet()
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = torch.relu(self.forward(batch))
        loss = self.criterion(output, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = torch.relu(self.forward(batch))
        loss = self.criterion(output, batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer