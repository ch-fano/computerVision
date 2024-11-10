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

