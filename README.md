# Computer Vision Project

The project consists of three main packages:
- *pipeline_utilities*: contains the files necessary for setting up and using the recognition pipeline.
- *support_utilities*: includes files for analyzing and visualizing results
- *yolov5*: contains the ultralytics yolov5 code and the trained models.

## Pipeline utilities

This package includes the following files: 
- *class_discovery*: tests the performance of the recognition pipeline.
- *cross_validation*: implements cross-validation after dataset creation. 
- *data_augmentation*: provides a function for applying data augmentation to images.
- *dataset_creation*: creates a dataset from a directory of images, organized by problem class.
- *recognition_pipeline*: defines the class used to recognize the problem classes in an image.

## Support utilities

This package includes the following files:
- *confusion_matrix*: generates a confusion matrix from provided data.
- *cross_validation_test*: tests the cross-validation splits.
- *delete_duplicates*: removes duplicate images from a directory if they exist in another directory.
- *measure_brightness*: contains functions for analyzing and visualizing the brightness of class images.

## Yolov5

This package is a clone of the https://github.com/ultralytics/yolov5 repository and is used to train models on the created dataset:
- trained models are located in the runs/train directory.
- YAML configuration files used during training are in the data directory.

The convention used for naming the trained models is as follows *ncC_neE[_DA]*:
- *nc* represents the number of classes.
- *ne* indicates the number of epochs.
- *DA* is added if data augmentation has been applied.
  
For example, 10C_40E indicates a model trained on 10 classes for 40 epochs.
All these models are trained on the dataset split into training, validation and test sets using the 80-10-10 division.

Models with names starting with *FIN_neE* are trained on a dataset divided only into training and validation sets, using a 85-15 split, with data augmentation applied to the training set.
