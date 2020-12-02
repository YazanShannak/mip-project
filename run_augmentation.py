import os
from src.data.augment import is_positive, augment_mode_1, augment_mode_2
from PIL import Image

data_dir = os.path.join(os.curdir, "data")
raw_dir = os.path.join(data_dir, "raw")
processed_dir = os.path.join(data_dir, "processed")

image_dir_name = "png_images"
mask_dir_name = "png_masks"

raw_images_dir = os.path.join(raw_dir, image_dir_name)
raw_masks_dir = os.path.join(raw_dir, mask_dir_name)

sample_image = os.path.join(raw_images_dir, "0_test_1_.png")
sample_mask = os.path.join(raw_masks_dir, "0_test_1_.png")

image, mask = Image.open(fp=sample_image), Image.open(fp=sample_mask)
image, mask = augment_mode_2(image, mask)
image.save(fp="sample_image.png", format="PNG")
mask.save(fp="sample_mask.png", format="PNG")
# for image_name in os.listdir(raw_images_dir):
#     if is_positive(name=image_name):
#         print(image_name)
#         image_path = os.path.join(raw_images_dir, image_name)
#         mask_path = os.path.join(raw_masks_dir, image_name)
#         image, mask = Image.open(fp=image_path), Image.open(fp=mask_path)
#         image, mask = augment_mode_2(image, mask)
#         image.save(fp="sample_image.png", format="PNG")
#         mask.save(fp="sample_mask.png", format="PNG")
#         break
