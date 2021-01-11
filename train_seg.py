import os
import torch
import pytorch_lightning as pl
from src.data.segmentation_loader import SegmentationLoader
from src.models.segmentation import SegmentationUnet
from src.models.resnet_backbone import ResNetBackBone
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


def parse_weights(state_dict):
    en_weights = {key[8:]: value for key, value in state_dict.items() if "encoder" in key}
    de_weights = {key[8:]: value for key, value in state_dict.items() if "decoder" in key}
    return en_weights, de_weights


data_dir = os.path.join(os.curdir, "data")
processed_dir = os.path.abspath(os.path.join(data_dir, "processed"))
checkpoint_path = os.path.join(os.curdir, "logs", "autoencoder", "version_2", "checkpoints", "epoch=0.ckpt")
images_datamodule = SegmentationLoader(data_dir=processed_dir, batch_size=2)

model_name = "unet"
version = "autoencoder-raw-2"
lr = 1e-4
gamma = 1
freeze_encoder = True

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
tensorboard_logger = TensorBoardLogger(save_dir="logs", name=model_name, version=version)
lr_logger = LearningRateMonitor(logging_interval='epoch')
early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, mode="min", patience=5)

pretrained_weights = torch.load(checkpoint_path)["state_dict"]
encoder_weights, decoder_weights = parse_weights(pretrained_weights)

# model = SegmentationUnet(lr=lr, gamma=gamma, freeze_encoder=freeze_encoder, encoder_weights=encoder_weights)
model = ResNetBackBone(lr=lr, gamma=gamma)

trainer = pl.Trainer(gpus=-1, logger=tensorboard_logger, max_epochs=50,
                     callbacks=[checkpoint_callback, lr_logger, early_stop])

if __name__ == "__main__":
    trainer = trainer.fit(model=model, datamodule=images_datamodule)
