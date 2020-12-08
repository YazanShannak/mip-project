import os
import shutil
from src.data.utils import is_train

data_dir = os.path.join(os.curdir, "data")
processed = os.path.join(data_dir, "processed")

source_dir = os.path.join(processed, "masks")
train_dir = os.path.join(processed, "train", "masks")
test_dir = os.path.join(processed, "test", "masks")
train_count = 0
test_count = 0
for file in os.listdir(source_dir):
    old_path = os.path.join(source_dir, file)
    new_path = os.path.join(train_dir, file) if is_train(file) else os.path.join(test_dir, file)
    shutil.move(src=old_path, dst=new_path)
