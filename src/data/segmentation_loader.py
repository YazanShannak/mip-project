import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from typing import Tuple


class SegmentationLoader(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64):
        super(SegmentationLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(data_dir=os.path.join(self.data_dir, "train"))
        self.test_dataset = SegmentationDataset(data_dir=os.path.join(self.data_dir, "test"))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str):
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.all_images = os.listdir(self.images_dir)
        self.transforms = Compose([
            Resize(size=(512, 512)),
            ToTensor()
        ])
        self.image_transforms = Compose([
            Resize(size=(512, 512)),
            ToTensor(),
            Normalize((0.4828,), (0.2488,))
        ])

    def __len__(self) -> int:
        return len(self.all_images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name)
        return self.image_transforms(Image.open(fp=image_path)), torch.where(
            self.transforms(Image.open(fp=mask_path)) > 0, 1, 0).float()
