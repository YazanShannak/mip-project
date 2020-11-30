mkdir $(pwd)/data
mkdir $(pwd)/data/raw

kaggle datasets download -p $(pwd)/data/raw --unzip vbookshelf/pneumothorax-chest-xray-images-and-masks
