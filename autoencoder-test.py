import os
from PIL import Image
from src.data.autoencoder_loader import AutoencoderDataLoader
from src.models.autoencoder import AutoEncoder

data_dir = os.path.join(os.curdir, "data")
processed_dir = os.path.abspath(os.path.join(data_dir, "processed"))

images_datamodule = AutoencoderDataLoader(data_dir=processed_dir, batch_size=20)
images_datamodule.setup()
checkpoint_path = os.path.join(os.curdir, "logs", "autoencoder", "version_5", "checkpoints", "epoch=0.ckpt")

model = AutoEncoder().load_from_checkpoint(checkpoint_path=checkpoint_path)

sample = next(iter(images_datamodule.val_dataloader()))
reconstruction = model(sample).detach().cpu().numpy()[13, 0, :, :] * 255
reconstruction = reconstruction.astype("uint8")
im = Image.fromarray(reconstruction, mode="L")
im.save("test2.png")
