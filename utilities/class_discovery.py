import os
import numpy as np
import torch
import cv2


def get_best_class(probabilities):
    """
    This function extract the label with the highest probability.
    
    :param probabilities: List of [label, probability].
    :return: The [label, probability] with the highest probability.
    """
    best = []
    for elem in probabilities:

        if len(best) == 0 or elem[1] > best[1]:
            best = elem

    # Return class name and with the best confidence
    return best


def yolo_detection(img_path):
    """
    This function process the image with the trained yolo model and returns the predicted class with
    the highest confidence.
    
    :param img_path: The path to the image to recognize.
    :return: The class with the highest probability.
    """
    
    # Insert the path to the yolov5 directory
    model_dir = '/yolov5'

    # Insert the path to the custom weights
    custom_weights = '/home/christofer/PycharmProjects/computerVision/yolov5/runs/train/exp24/weights/best.pt'

    model = torch.hub.load(model_dir,'custom', source='local', path=custom_weights, force_reload=True)
    model.conf = 0.25  # confidence threshold
    model.iou = 0.45   # IoU threshold
    model.classes = None  # filter by class
    model.img_size = 640  # set image size to 640x640 pixels

    # Perform inference
    results = model(img_path)

    # results.print()
    # results.save()   # Save results to disk (runs in YOLOv5/runs/detect)
    # results.show()     # Shows the image and the bbox

    # Access detailed results if needed
    df = results.pandas().xyxy[0]  # Get bounding boxes as pandas DataFrame
    #print(df)

    # Return class name and with the best confidence
    return get_best_class(df[['name', 'confidence']].values.tolist())


def get_max_value(dtype):
    """
    This function returns the maximum brightness value of a given image type.
    
    :param dtype: The image type
    :return: The maximum brightness value the given image type.
    """
    
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return 1.0  # Typically for floating point images
    else:
        raise ValueError("Unsupported dtype")


def get_brightness(img_path):
    """
    This function returns the brightness expressed form 0 to 1 of a given image.
    
    :param img_path: The path to the image to calculate the brightness.
    :return: The brightness of the image.
    """
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Warning: Image not found or could not be loaded: {img_path}")
        return None

    # Determine the max possible value for the image's dtype
    max_val = get_max_value(image.dtype)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.mean(gray_image) / max_val


def is_illumination(img_path):
    """
    This function returns the probability of the given image to be an instance of the 'illumination' class.
    
    :param img_path: The image to compute the probability.
    :return: The probability of the image to be an instance of the 'illumination' class.
    """
    
    threshold = 0.30
    brightness = get_brightness(img_path)

    if brightness is None:
        return 0  # Assuming no illumination class if the image can't be processed

    if brightness <= threshold:
        return 1 - (brightness/threshold)

    return 0

def recognition_pipeline(image_path):
    """
    This function implements the recognition pipeline of an image, it extracts the probabilities
    of the image to be an instance of each class.
    
    :param image_path: The image to recognize.
    :return: The class with the highest probability and the list of the classes of which the image could be an instance.
    """
    
    classes = []

    illumination_prob = is_illumination(image_path)

    if illumination_prob:
        classes.append(['illuminazione_brightness', illumination_prob])

    if illumination_prob <= 0.50:
        label = yolo_detection(image_path)

        if len(label) != 0:
            classes.append(label)

    return get_best_class(classes), classes

if __name__ == '__main__':

    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']

    test_dir = ''
    dir_classes = {'undefined': 0}
    tot_imgs = 0

    for image_path in os.listdir(test_dir):

        full_path = os.path.join(test_dir, image_path)

        if os.path.splitext(full_path)[1].lower() in valid_image_extensions:
            print(f"\nProcessing image: {full_path}")
            tot_imgs += 1

            best, classes = recognition_pipeline(full_path)

            if len(best) == 0:
                dir_classes['undefined'] +=1
            elif best[0] in dir_classes:
                dir_classes[best[0]] += 1
            else:
                dir_classes[best[0]] = 1
        else:
            print(f"Skipping non-image file: {full_path}")


    print(f'Total images: {tot_imgs}')
    print('Predicted results: ')
    for key, value in dir_classes.items():
        print(f'- {key}: {value}')

    #best, classes = test_img(image_path)
    #print('Classes: ', classes)
    #print('Best: ', best)
