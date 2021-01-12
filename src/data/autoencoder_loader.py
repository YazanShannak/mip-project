import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


class AutoencoderDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, mode: str = "images"):
        super(AutoencoderDataLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode

    def setup(self, stage=None):
        self.train_dataset = AutoencoderDataset(data_dir=os.path.join(self.data_dir, "train"), mode=self.mode)
        self.test_dataset = AutoencoderDataset(data_dir=os.path.join(self.data_dir, "test"), mode=self.mode)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)


class AutoencoderDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = "images"):
        self.data_dir = os.path.join(data_dir, "images") if mode == "images" else os.path.join(data_dir, "masks")
        self.all_images = os.listdir(self.data_dir)
        basic_transforms = [Resize(size=(512, 512)), ToTensor()]
        self.transforms = Compose(
            [*basic_transforms, Normalize((0.4828,), (0.2488,))]) if mode == "images" else Compose(basic_transforms)

    def __len__(self) -> int:
        return int(len(self.all_images))

    def __getitem__(self, index: int) -> torch.Tensor:
        image_name = self.all_images[index]
        path = os.path.join(self.data_dir, image_name)
        return self.transforms(Image.open(fp=path))
