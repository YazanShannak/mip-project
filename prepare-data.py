import os
import shutil
from tqdm import tqdm
from src.data.augment import augment_mode_1, augment_mode_2
from src.data.utils import is_positive, change_id_name, get_id, is_train
from PIL import Image

data_dir = os.path.join(os.curdir, "data")
raw_dir = os.path.join(data_dir, "raw", "siim-acr-pneumothorax")

processed_dir = os.path.join(data_dir, "processed")
train_dir = os.path.join(processed_dir, "train")
test_dir = os.path.join(processed_dir, "test")
try:
    os.mkdir(processed_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(os.path.join(train_dir, "images"))
    os.mkdir(os.path.join(train_dir, "masks"))
    os.mkdir(os.path.join(test_dir, "images"))
    os.mkdir(os.path.join(test_dir, "masks"))
except:
    pass

raw_images_dir = os.path.join(raw_dir, "png_images")
raw_masks_dir = os.path.join(raw_dir, "png_masks")

for image_name in tqdm(os.listdir(raw_images_dir), desc="Augmenting and copying data"):
    processed_images_dir = os.path.join(train_dir if is_train(image_name) else test_dir, "images")
    processed_masks_dir = os.path.join(train_dir if is_train(image_name) else test_dir, "masks")
    image_path = os.path.join(raw_images_dir, image_name)
    mask_path = os.path.join(raw_masks_dir, image_name)
    image, mask = Image.open(fp=image_path), Image.open(fp=mask_path)

    if is_positive(name=image_name):
        aug_image_1, aug_mask_1 = augment_mode_1(image, mask)
        aug_image_2, aug_mask_2 = augment_mode_2(image, mask)

        old_id = get_id(image_name)
        aug_1_id = change_id_name(original_name=image_name, new_id=old_id + "aug1")
        aug_2_id = change_id_name(original_name=image_name, new_id=old_id + "aug2")

        aug_image_1.save(os.path.join(processed_images_dir, aug_1_id), "PNG")
        aug_mask_1.save(os.path.join(processed_masks_dir, aug_1_id), "PNG")

        aug_image_2.save(os.path.join(processed_images_dir, aug_2_id), "PNG")
        aug_mask_2.save(os.path.join(processed_masks_dir, aug_2_id), "PNG")

    shutil.copyfile(src=image_path, dst=os.path.join(processed_images_dir, image_name))
    shutil.copyfile(src=mask_path, dst=os.path.join(processed_masks_dir, image_name))
