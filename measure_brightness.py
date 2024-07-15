import numpy as np
import cv2
import os

from class_discovery import get_brightness, is_lighting


def get_class_brightness(class_path):
    brightness = []

    for img in os.listdir(class_path):

        brightness.append(get_brightness(os.path.join(class_path, img)))

    print(brightness)
    print('min: ', min(brightness))
    print('max: ', max(brightness))
    print('mean: ', sum(brightness) / len(brightness))
    print('median: ', np.median(brightness))


def test_threshold(class_path, threshold):

    tot_elem = 0
    passed_elem = 0

    for img in os.listdir(class_path):

        brightness = get_brightness(os.path.join(class_path, img))

        tot_elem += 1

        if brightness <= threshold:
            passed_elem += 1

    print('Total elements: ', tot_elem)
    print('Passed elements: ', passed_elem)


def test_probability(class_path):

    prob = []

    for img in os.listdir(class_path):
        prob.append(is_lighting(os.path.join(class_path, img)))

    print('Probability mean: ', sum(prob) / len(prob))


if __name__ == '__main__':

    # insert here the path to the directory of the class
    c_path = ''

    # get_class_brightness(c_path)
    # test_threshold(c_path, 0.34)
