import torch
import pytorch_lightning as pl
from src.models.unet import UnetEncoder, UnetDecoder
from torch import nn


class AutoEncoder(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, gamma: float = 0.5):
        super(AutoEncoder, self).__init__()
        self.save_hyperparameters()
        self.encoder = UnetEncoder()
        self.decoder = UnetDecoder()
        self.lr = lr
        self.gamma = gamma
        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        output = self.decoder(*encoder_outputs)
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        batch = (batch * 0.2488) + 0.4828
        loss = self.criterion(output, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        batch = (batch * 0.2488) + 0.4828
        loss = self.criterion(output, batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        random_index = torch.randint(low=0, high=len(batch) - 1, size=(1,)).item().cpu()
        self.logger.experiment.add_image("Reconstructed Image", output[random_index], self.global_step,
                                         dataformats="CHW")
        self.logger.experiment.add_image("Actual Image", batch[random_index], self.global_step,
                                         dataformats="CHW")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
