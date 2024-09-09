from time import sleep

import numpy as np
import os

from torch.nn.functional import threshold
from tqdm import tqdm
from class_discovery import get_brightness, is_lighting


def extract_images(class_path):
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
    imgs = []

    # extract the images
    for f in os.listdir(class_path):
        # check if it is an image
        if os.path.splitext(f)[1].lower() in valid_image_extensions:
            imgs.append(os.path.join(class_path, f))

    return imgs

def get_class_brightness(class_path):
    brightness = []
    imgs = extract_images(class_path)

    print('Class directory: ' + class_path)

    # evaluate the brightness
    for img in tqdm(imgs, desc="Evaluating the class brightness"):
        brightness.append(get_brightness(img))

    print(brightness)
    print('Min: ', min(brightness))
    print('Max: ', max(brightness))
    print('Mean: ', sum(brightness) / len(brightness))
    print('Median: ', np.median(brightness))


def test_threshold(class_path, threshold):
    '''
    This class tests the threshold and evaluate the number of images with brightness below it
    '''

    imgs = extract_images(class_path)
    passed_elem = 0

    print('\nClass directory: ' + class_path)
    print('Testing the threshold ' + str(threshold) + '...')

    for img in tqdm(imgs, desc="Testing the class images"):
        brightness = get_brightness(img)

        if brightness <= threshold:
            passed_elem += 1

    print('Class total elements: ', len(imgs))
    print('Class elements under the threshold: ', passed_elem)

    return len(imgs), passed_elem



def test_probability(class_path):
    '''
    This class evaluates the mean probability of the class to be labeled as the class "illuminazione"
    '''

    imgs = extract_images(class_path)
    prob = []

    print('Class directory: ' + class_path)

    for img in tqdm(imgs, desc="Evaluating the probability of each image"):
        prob.append(is_lighting(img))

    print('Probability mean: ', sum(prob) / len(prob))

def test_classes_threshold(base_dir, subdir_list, threshold):

    num_classes = len(subdir_list)

    tot_elem = 0
    tot_under_t = 0

    for n,subdir in enumerate(subdir_list):
        print(f'\nTesting subdir {n}/{num_classes}')

        dir_elem, dir_under_t = test_threshold(os.path.join(base_dir, subdir), threshold)
        tot_elem += dir_elem
        tot_under_t += dir_under_t

    print('\n---------------------------------------')
    print('Total elements: ', tot_elem)
    print('Elements under the threshold: ', tot_under_t)


if __name__ == '__main__':

    # Insert the path to the directory of the class
    c_path = '/home/christofer/Desktop/temp'
    threshold = 0.30

    base_folder_name = '/home/christofer/Desktop/cv_images'
    subdir_l = ['1_strada_buca',
                '4_semaforo_non_funzionante',
                '11_segnaletica_danneggiata',
                '14_graffiti',
                '20_veicolo_abbandonato',
                '21_bicicletta_abbandonata',
                #'22_strada_al_buio',
                '27_deiezioni_canine',
                '156_siringa_abbandonata',
                '159_rifiuti_abbandonati']

    test_classes_threshold(base_folder_name, subdir_l, threshold)
    test_threshold(c_path, threshold)
