TODO:

- [x] Preprocessing Operation
  - [x] separate the positive and negative cases in in a separated folders for both images and masks
  - [x] perform 2 augmentation operations on both images and masks of the positive class labeled with 1 and perform the following
    1- flip all positive  images  around the vertical axis
    2- perform a random rotations in range of (1 to 10)  degrees in the positive side for half of the positive images and (-1-10) for the second half
    3-Keep in mind that the consistency between each image and its corresponding mask, ie , the same augmentation is performed on both image and mask especially in case of rotation.
  - [x] regroup all images from both cases (+ve and -ve)
  - [x] resize all images into 512 * 512
  - [x] normalize the images and masks
  - [x] split the data into training and validation datasets 



have a nice day