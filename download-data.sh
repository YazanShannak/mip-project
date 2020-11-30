mkdir $(pwd)/data
mkdir $(pwd)/data/raw

kaggle datasets download -p $(pwd)/data/raw --unzip iafoss/siimacr-pneumothorax-segmentation-data-512
