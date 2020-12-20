import os
import pytorch_lightning as pl
from src.data.segmentation_loader import SegmentationLoader
from src.models.segmentation import SegmentationUnet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

data_dir = os.path.join(os.curdir, "data")
processed_dir = os.path.abspath(os.path.join(data_dir, "processed"))
checkpoint_path = os.path.join(os.curdir, "logs", "autoencoder", "version_5", "checkpoints", "epoch=3.ckpt")
images_datamodule = SegmentationLoader(data_dir=processed_dir, batch_size=2)

model_name = "unet"
version = 7
lr = 1e-2

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
tensorboard_logger = TensorBoardLogger(save_dir="logs", name=model_name, version=version)

# model = SegmentationUnet.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=True, lr=lr, gamma=0.1)
model = SegmentationUnet(lr=lr, gamma=0.5)

trainer = pl.Trainer(gpus=-1, logger=tensorboard_logger, max_epochs=35, callbacks=[checkpoint_callback])
trainer = trainer.fit(model=model, datamodule=images_datamodule)
