# Medical Image Processing Project

<<<<<<< HEAD
- [x] Preprocessing Operation
  - [x] separate the positive and negative cases in in a separated folders for both images and masks
  - [x] perform 2 augmentation operations on both images and masks of the positive class labeled with 1 and perform the following
    1- flip all positive  images  around the vertical axis
    2- perform a random rotations in range of (1 to 10)  degrees in the positive side for half of the positive images and (-1-10) for the second half
    3-Keep in mind that the consistency between each image and its corresponding mask, ie , the same augmentation is performed on both image and mask especially in case of rotation.
  - [x] regroup all images from both cases (+ve and -ve)
  - [x] resize all images into 512 * 512
  - [x] normalize the images and masks
  - [x] Split the data into training and validation datasets
## Authors
>>>>>>> 3e2f66c00f3fefc414580b6352d7c4701be562ab

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
