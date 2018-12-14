# Satellite Image Classifier

Script to identify satellite images that contain buildings using VGG-16 CNN.

The model receives as input the satellite images and the corresponding identification of wether or not it contains a building, in order to be trained.
For each image in the test set, a classification is made by identifying if it contains a building or not.

## Training

The model is trained for 10 epochs with the binary crossentropy loss function and using image augmentation
in order to generate more combinations of images to make the model more robust to changes.

After 50 epochs the calculated accuracy is about 87%.

Using a GeForce GTX 1060 3GB, the training phase takes nearly 3 hours for a training set of 5000 images and a batch size of 16.

## Dependencies

This script depends on the following libraries:

* Tensorflow
* Keras >= 1.0

The code should be compatible with Python versions 2.7-3.5.


## Built With

* [VGG-16](https://arxiv.org/abs/1409.1556) - The convolutional Neural Network architecture used.
