# Medical Image Processing Project

## Authors

Yazan Shannak  - ywshannak19@cit.just.edu.jo

Saeed Shurrab  -  sashurrab18@cit.just.edu.jo



## Dataset

The original dataset can be found on [Kaggle](https://www.kaggle.com/vbookshelf/pneumothorax-chest-xray-images-and-masks) 

The processed dataset (Our work) is available on [S3](https://yazansh-public.s3.amazonaws.com/mip-data.zip), and can be downloaded and extracted with  running as executable ./download-data.sh ( make sure permissions are provided for the script)  



## Paper

Paper can be found in the paper/ directory, the source LateX code alongside the compiled PDF version.



## Code Structure

Two main files to run an expirement:

1. train-autoencoder.py
2. train-segmentation.py



Models, dataloaders, and other utils can be found at the src/ directory



## Running the Code

It's advisable to run the code via Docker

1. Build the image with 

   `docker image build -t mip-project:latest .`

2. See  https://github.com/YazanShannak/docker-pytorch-lightning for more details on running this image
