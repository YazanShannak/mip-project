import os
import pytorch_lightning as pl
from src.data.segmentation_loader import SegmentationLoader
from src.models.segmentation import SegmentationUnet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

data_dir = os.path.join(os.curdir, "data")
processed_dir = os.path.abspath(os.path.join(data_dir, "processed"))
checkpoint_path = os.path.join(os.curdir, "logs", "autoencoder", "version_1", "checkpoints", "epoch=19.ckpt")
images_datamodule = SegmentationLoader(data_dir=processed_dir, batch_size=2)

model_name = "unet"
version = 2
lr = 1e-4
gamma = 0.75

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
tensorboard_logger = TensorBoardLogger(save_dir="logs", name=model_name, version=version)
lr_logger = LearningRateMonitor(logging_interval='epoch')

model = SegmentationUnet.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=True, lr=lr, gamma=gamma)
# model = SegmentationUnet(lr=lr, gamma=gamma)

trainer = pl.Trainer(gpus=-1, logger=tensorboard_logger, max_epochs=35, callbacks=[checkpoint_callback, lr_logger])
trainer = trainer.fit(model=model, datamodule=images_datamodule)
