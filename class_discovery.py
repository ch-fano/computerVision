import numpy as np
import torch
import cv2


def yolo_detection(img_path):

    # insert the path to the yolov5 directory
    model_dir = ''

    # insert the path to the custom weights
    custom_weights = ''

    model = torch.hub.load(model_dir,'custom', source='local', path=custom_weights, force_reload=True)
    model.conf = 0.25  # confidence threshold
    model.iou = 0.45   # IoU threshold
    model.classes = None  # filter by class
    model.img_size = 640  # set image size to 640x640 pixels

    # Perform inference
    results = model(img_path)

    # Print results
    results.print()  # Print results to console
    # results.save()   # Save results to disk (runs in YOLOv5/runs/detect)
    results.show()   # Show results

    # Access detailed results if needed
    df = results.pandas().xyxy[0]  # Get bounding boxes as pandas DataFrame
    print(df)


def get_max_value(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return 1.0  # Typically for floating point images
    else:
        raise ValueError("Unsupported dtype")


def get_brightness(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Determine the max possible value for the image's dtype
    max_val = get_max_value(image.dtype)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.mean(gray_image) / max_val


def is_lighting(img_path):

    threshold = 0.34
    brightness = get_brightness(img_path)

    if brightness <= threshold:
        return 1 - (brightness/threshold)

    return 0


if __name__ == '__main__':

    image_path = ''
    labels = []

    lighting_prob = is_lighting(image_path)

    # TO DO: insert a probability threshold
    if lighting_prob:
        labels.append('illuminazione')
    else:

        # TO DO: extract the labels from yolo
        yolo_detection(image_path)
