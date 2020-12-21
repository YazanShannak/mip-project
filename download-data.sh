mkdir $(pwd)/data
mkdir $(pwd)/data/processed

#kaggle datasets download -p $(pwd)/data/raw --unzip vbookshelf/pneumothorax-chest-xray-images-and-masks

wget  https://yazansh-public.s3.amazonaws.com/mip-data.zip -P $(pwd)/data/processed
unzip data/processed/mip-data.zip
