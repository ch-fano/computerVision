import numpy as np
import pickle
import os
from recognition_pipeline import RecognitionPipeline as rp
from tqdm import tqdm

import matplotlib.pyplot as plt


def extract_images(class_path):
    """
    This function scans a directory and returns a list of paths for all image files
    that have common image file extensions.

    :param class_path: The directory to search for image files.
    :return: A list of full paths to all image files in the directory.
    """

    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
    imgs = []

    # extract the images
    for f in os.listdir(class_path):
        # check if it is an image
        if os.path.splitext(f)[1].lower() in valid_image_extensions:
            imgs.append(os.path.join(class_path, f))

    return imgs

def get_class_brightness(class_path):
    """
    This function evaluate the brightness of all image files in the specified directory and
    returns the minimum, maximum, mean and median brightness.

    :param class_path: The directory to compute the metrics.
    """

    brightness = []
    imgs = extract_images(class_path)

    print('Class directory: ' + class_path)

    # evaluate the brightness
    for img in tqdm(imgs, desc="Evaluating the class brightness"):
        brightness.append(rp.get_brightness(img))

    print(brightness)
    print('Min: ', min(brightness))
    print('Max: ', max(brightness))
    print('Mean: ', sum(brightness) / len(brightness))
    print('Median: ', np.median(brightness))


def test_class_threshold(class_path, threshold):
    """
    This function tests the threshold and evaluate the number of images with brightness below it
    for all the images in the directory.

    :param class_path: The directory to test.
    :param threshold: The threshold to test.
    :return: The total number of images and the number of images with brightness below the threshold.
    """

    imgs = extract_images(class_path)
    passed_elem = 0

    print('\nClass directory: ' + class_path)
    print('Testing the threshold ' + str(threshold) + '...')

    for img in tqdm(imgs, desc="Testing the class images"):
        brightness = rp.get_brightness(img)

        if brightness <= threshold:
            passed_elem += 1

    print('Class total elements: ', len(imgs))
    print('Class elements under the threshold: ', passed_elem)

    return len(imgs), passed_elem

def test_classes_threshold(base_dir, subdir_list, threshold):
    """
    This function tests the threshold and evaluate the number of images with brightness below it for all
    the specified directories.

    :param base_dir: The common base directory.
    :param subdir_list: The list of subdirectories to test.
    :param threshold: The threshold to test.
    """

    num_classes = len(subdir_list)

    tot_elem = 0
    tot_under_t = 0

    for n,subdir in enumerate(subdir_list):
        print(f'\nTesting subdir {n+1}/{num_classes}')

        dir_elem, dir_under_t = test_class_threshold(os.path.join(base_dir, subdir), threshold)
        tot_elem += dir_elem
        tot_under_t += dir_under_t

    print('\n---------------------------------------')
    print('Total elements: ', tot_elem)
    print('Elements under the threshold: ', tot_under_t)


def create_brightness_pickle(base_dir, classes_subdir_list, illumination_path_dir, pickle_filename):
    """
    This function compute the brightness of the images in the specified directories and
    writes the result in the pickle file.

    :param base_dir: The common base directory.
    :param classes_subdir_list: The list of subdirectories to test.
    :param illumination_path_dir: The path to the directory of the 'illumination' class.
    :param pickle_filename: The name of the pickle file.
    """
    num_classes = len(classes_subdir_list)

    brightness_dict = {
        'illumination': [],
        'other_classes' : [],
    }

    # Extract the brightness of the illumination class
    print(f'\nClass directory: {illumination_path_dir}')

    imgs = extract_images(illumination_path_dir)
    for img in tqdm(imgs, desc="Computing the class images' brightness"):
        brightness_dict['illumination'].append(rp.get_brightness(img))


    # Extract the other classes brightness
    for n,subdir in enumerate(classes_subdir_list):
        print(f'\nEvaluating subdir {n+1}/{num_classes}')
        print(f'\nClass directory: {os.path.join(base_dir, subdir)}' )

        imgs = extract_images(os.path.join(base_dir, subdir))
        for img in tqdm(imgs, desc="Computing the class images' brightness"):
            brightness_dict['other_classes'].append(rp.get_brightness(img))

    print('Creating pickle file...')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(brightness_dict, f)

    print('Pickle file created.')

def brightness_graph(pickle_filename):
    """
    This function extract the classes brightness from the pickle file, tests the classes with increasing
    threshold from 0 to 1 and plots the result in 'brightness_chart.png'.

    :param pickle_filename: The name of the pickle file.
    """

    with open(pickle_filename, 'rb') as f:
        brightness_dict = pickle.load(f)

    tot_other_classes = len(brightness_dict['other_classes'])
    tot_illumination = len(brightness_dict['illumination'])

    other_classes = []
    illumination = []

    values = np.arange(0, 1.05, 0.05)  # The stop is 1.05 to include 1
    threshold_list = list(values)

    for threshold in threshold_list:
        below_elem = 0
        for brightness in brightness_dict['illumination']:
            if brightness <= threshold:
                below_elem += 1
        correctness = below_elem / tot_illumination
        illumination.append(correctness)

        below_elem = 0
        for brightness in brightness_dict['other_classes']:
            if brightness <= threshold:
                below_elem += 1
        correctness = (tot_other_classes - below_elem) / tot_other_classes
        other_classes.append(correctness)

    plt.plot(threshold_list, illumination, marker='o', color='blue', label='Illuminazione')
    plt.plot(threshold_list, other_classes, marker='o', color='red', label='Altre classi')

    plt.xlabel('Soglia di luminosità')
    plt.ylabel('Correttezza')
    plt.title('Performance - Soglia Luminosità')
    #plt.grid(True)

    plt.xticks(threshold_list) # Set the step to 0.05 for the X-axis
    plt.yticks(threshold_list) # Set the step to 0.05 for the Y-axis

    # Rotate the x-axis labels to prevent overlapping
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    # Add a legend to distinguish the two lines
    plt.legend()
    plt.savefig('brightness_chart.png')


if __name__ == '__main__':

    # Insert the path to the directory of the class
    illumination_path = '/home/christofer/Desktop/illumination/'
    base_folder_name = '/home/christofer/Desktop/cv_images'
    subdir_l = [
        '1_strada_buca',
        '4_semaforo_non_funzionante',
        '11_segnaletica_danneggiata',
        '14_graffiti',
        '20_veicolo_abbandonato',
        '21_bicicletta_abbandonata',
        #'22_strada_al_buio',
        '27_deiezioni_canine',
        '47_scuola',
        '156_siringa_abbandonata',
        '159_rifiuti_abbandonati'
    ]
    p_filename = 'files/brightness.pickle'
    threshold = 0.30

    # test_classes_threshold(base_folder_name, subdir_l, threshold)
    # create_brightness_pickle(base_folder_name, subdir_l, illumination_path, p_filename)
    # brightness_graph(p_filename)