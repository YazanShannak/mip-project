import os
import pytorch_lightning as pl
from src.data.autoencoder_loader import AutoencoderDataLoader
from src.models.autoencoder import AutoEncoder
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

data_dir = os.path.join(os.curdir, "data")
processed_dir = os.path.abspath(os.path.join(data_dir, "processed"))

images_datamodule = AutoencoderDataLoader(data_dir=processed_dir, batch_size=4)

model_name = "autoencoder"
version = 1
lr = 1e-4

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

tensorboard_logger = TensorBoardLogger(save_dir="logs", name=model_name, version=version)

model = AutoEncoder(lr=lr)

trainer = pl.Trainer(gpus=-1, logger=tensorboard_logger, max_epochs=35, callbacks=[checkpoint_callback])
trainer = trainer.fit(model=model, datamodule=images_datamodule)
