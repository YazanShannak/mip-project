import os
import pytorch_lightning as pl
from src.data.autoencoder_loader import AutoencoderDataLoader
from src.models.autoencoder import AutoEncoder
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

data_dir = os.path.join(os.curdir, "data")
processed_dir = os.path.abspath(os.path.join(data_dir, "processed"))

mode = "masks"

images_datamodule = AutoencoderDataLoader(data_dir=processed_dir, batch_size=16, mode=mode)

model_name = "autoencoder"
version = 3
lr = 1e-4
gamma = 1

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
tensorboard_logger = TensorBoardLogger(save_dir="logs", name=model_name, version=version)
lr_logger = LearningRateMonitor(logging_interval='epoch')
early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, mode="min", patience=5)

model = AutoEncoder(lr=lr, gamma=gamma)

trainer = pl.Trainer(gpus=-1, logger=tensorboard_logger, max_epochs=50,
                     callbacks=[checkpoint_callback, lr_logger, early_stop],
                     precision=16)

if __name__ == "__main__":
    trainer = trainer.fit(model=model, datamodule=images_datamodule)
