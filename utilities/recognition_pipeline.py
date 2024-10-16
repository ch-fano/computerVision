import numpy as np
import torch
import cv2

class RecognitionPipeline:
    def __init__(self, yolov5_dir, custom_weights):
        print("Initializing the yolov5 model of the recognition pipeline")

        self.model = torch.hub.load(yolov5_dir, 'custom', source='local', path=custom_weights, force_reload=True)
        self.model.conf = 0.01  # confidence threshold (before was 0.25)
        self.model.iou = 0.45  # IoU threshold
        self.model.classes = None  # filter by class
        self.model.img_size = 640  # set image size to 640x640 pixels

        self.classes_dict = self.model.names

    @staticmethod
    def get_imgtype_max_value(dtype):
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

    @staticmethod
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
        max_val = RecognitionPipeline.get_imgtype_max_value(image.dtype)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return np.mean(gray_image) / max_val

    @staticmethod
    def is_illumination(img_path, threshold):
        """
        This function returns the probability of the given image to be an instance of the 'illumination' class.

        :param img_path: The image to compute the probability.
        :param threshold: The brightness threshold, if the brightness of the image is under the threshold the picture could
                          be an instance of the 'illumination' class.
        :return: The probability of the image to be an instance of the 'illumination' class.
        """

        brightness = RecognitionPipeline.get_brightness(img_path)

        if brightness is None:
            return 0  # Assuming no illumination class if the image can't be processed

        if brightness <= threshold:
            return 1 - (brightness / threshold)

        return 0

    def yolo_detection(self, img_path):
        """
        This function process the image with the trained yolo model and returns the predicted class with
        the highest confidence.

        :param img_path: The path to the image to recognize.
        :return: List of [class, probability] of which the image could be an instance.
        """

        # Perform inference
        results = self.model(img_path)

        # results.print()
        # results.save()     # Save results to disk (runs in YOLOv5/runs/detect)
        # results.show()     # Shows the image and the bbox

        # Access detailed results if needed
        df = results.pandas().xyxy[0]  # Get bounding boxes as pandas DataFrame
        # print(df)

        # Return class name and with the best confidence
        return df[['name', 'confidence']].values.tolist()

    def recognize(self, image_path, brightness_threshold=0.30):
        """
        This function implements the recognition pipeline of an image, it extracts the probabilities
        of the image to be an instance of each class.

        :param image_path: The image to recognize.
        :param brightness_threshold: The brightness threshold, if the brightness of an image is under the threshold the picture could
                          be an instance of the 'illumination' class.
        :return: The class with the highest probability and the list of the classes of which the image could be an instance.
        """

        classes = self.yolo_detection(image_path)

        illumination_prob = self.is_illumination(image_path, brightness_threshold)

        if illumination_prob:
            classes.append(['illuminazione', illumination_prob])

        classes.sort(key=lambda x: x[1], reverse=True)
        return classes
