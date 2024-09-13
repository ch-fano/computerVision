import os
import shutil
import random
from sklearn.model_selection import train_test_split
from data_augmentation import data_augmentation as DA



def create_local_copy(src_path):
    """
    This function creates a local copy the specified directory in the project directory.

    :param src_path: The directory to copy.
    """
    if os.path.isdir(src_path):
        shutil.copytree(src_path, os.path.join(os.getcwd(), img_folder_name))
        print('---- Created local copy ----')


# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    """
    This function moves the specified files to the specified folder.

    :param list_of_files: The list of files to move.
    :param destination_folder: The destination folder.
    """
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except Exception as e:
            print(f'Error: {e}')
            print(f'File which caused the error: {f}')

def create_dataset_folders():
    """
    This function creates the dataset folders of the images and labels.
    """

    # Create the datasets directory
    parent_dir_datasets = os.path.join(os.getcwd(), 'datasets')
    os.mkdir(parent_dir_datasets)

    parent_dir_images = os.path.join(os.getcwd(), 'datasets', 'images')
    parent_dir_labels = os.path.join(os.getcwd(), 'datasets', 'labels')

    # Create the parent directories
    os.mkdir(parent_dir_images)
    os.mkdir(parent_dir_labels)

    # Create the subdirectories
    for new_dir in ['train', 'val', 'test']:
        # Create directory for images
        path = os.path.join(parent_dir_images, new_dir)
        os.mkdir(path)

        # Create directory for labels
        path = os.path.join(parent_dir_labels, new_dir)
        os.mkdir(path)

    print('---- Created images and labels directories ----')


def remove_dataset_folders(dir_list):
    """
    This function removes the specified folders in the current directory.

    :param dir_list: The list of folders to remove.
    """

    for d in dir_list:
        if os.path.isdir(d):
            shutil.rmtree(os.path.join(os.getcwd(), d), ignore_errors=False, onerror=None)

    print('---- Removed old folders ----')


def split_dataset():
    """
    This function randomly splits the dataset of images and labels into train, test and validation sets.

    """

    # Read images and annotations
    images = []
    annotations = []
    for f in os.listdir(tmp_folder_name):
        if f[-3:] == 'txt':
            annotations.append(os.path.join(tmp_folder_name, f))
        else:
            image_path = os.path.join(tmp_folder_name, f)

            if os.path.exists(os.path.splitext(image_path)[0]+'.txt'):
                images.append(image_path)
            else:
                print(f'No annotation for the image: {image_path}')

    images.sort()
    annotations.sort()

    # Split the dataset into train-valid-test splits
    random_seed = random.randint(0, 10000)
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations,
                                                                                    test_size=0.2,
                                                                                    random_state=random_seed)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                                  test_size=0.5,
                                                                                  random_state=random_seed)
    create_dataset_folders()

    # Move the splits into their folders
    move_files_to_folder(train_images, os.path.join('datasets', 'images', 'train'))
    move_files_to_folder(val_images, os.path.join('datasets', 'images', 'val'))
    move_files_to_folder(test_images, os.path.join('datasets','images','test'))
    move_files_to_folder(train_annotations, os.path.join('datasets', 'labels', 'train'))
    move_files_to_folder(val_annotations, os.path.join('datasets','labels','val'))
    move_files_to_folder(test_annotations, os.path.join('datasets','labels','test'))

    remove_dataset_folders([tmp_folder_name, img_folder_name])
    print('---- Dataset successfully splitted ----')


def apply_data_augmentation(dir_name, subdir_list, recursive=False):
    """
    This function apply data augmentation to the specified directories. At the end each directory has the same
    number of images.

    :param dir_name: The common base directory.
    :param subdir_list: The subdirectories with the images to augment.
    :param recursive: If 'True' it applies the data augmentation also on the images already augmented .
    """

    tot_files = []
    for subdir in subdir_list:
        path = os.path.join(dir_name, subdir)
        tot_files.append(len([name for name in os.listdir(path)]) // 2)

    i = 0
    max_num = max(tot_files) + max(tot_files) // 2 # 3/2 of the max_number

    # Extract the images from the subdirectory
    for subdir in subdir_list:
        path = os.path.join(dir_name, subdir)
        imgs = [os.path.join(path, f) for f in os.listdir(path) if f[-3:] != 'txt' and f != '.DS_Store']

        # Create new images to have the same number of pictures for each category
        while tot_files[i] < max_num:
            img_path = random.choice(imgs)
            img_root_ext = os.path.splitext(img_path)
            label_path = img_root_ext[0] + '.txt'

            new_img = DA(img_path, label_path)
            tot_files[i] = tot_files[i] + 1
            if recursive:
                imgs.append(new_img)

        i = i + 1

    print('---- Data augmentation ----')


def create_dataset(dir_name, subdir_list, augment=False, recursive=False):
    """
    This function moves all the images and labels in the specified subdirectories in a unique folder.

    :param dir_name: The common base directory.
    :param subdir_list: The subdirectories with the images and labels to move.
    :param augment: If 'True' it applies the data augmentation on the images.
    :param recursive: If 'True' it applies the data augmentation also on the images already augmented.
    :return:
    """

    if augment:
        apply_data_augmentation(dir_name, subdir_list, recursive)

    if os.path.exists(os.path.join(os.getcwd(), 'classes.txt')):
        os.remove(os.path.join(os.getcwd(), 'classes.txt'))

    shutil.move(os.path.join(dir_name, 'classes.txt'), os.getcwd())

    os.chdir(dir_name)
    files = []

    # For each directory extract the files and create a list of them
    for subdir in subdir_list:
        path = os.path.join(dir_name, subdir)
        files += [os.path.join(path, f) for f in os.listdir(subdir) if f != 'classes.txt']

    os.chdir('..')

    # Create the new temporary folder and move files on it
    os.mkdir(os.path.join(os.getcwd(), tmp_folder_name))
    move_files_to_folder(files, tmp_folder_name)

    print('---- Created the dataset ----')


if __name__ == '__main__':
    img_folder_name = 'img_copy'
    tmp_folder_name = 'tmp_folder'
    base_folder_name = '/home/christofer/Desktop/cv_images'  # Your path to the photo
    subdir_l = ['1_strada_buca',
                '4_semaforo_non_funzionante',
                '11_segnaletica_danneggiata',
                '14_graffiti',
                '20_veicolo_abbandonato',
                '21_bicicletta_abbandonata',
                '22_strada_al_buio',
                '27_deiezioni_canine',
                '156_siringa_abbandonata',
                '159_rifiuti_abbandonati']  # Add your subdir

    remove_dataset_folders([tmp_folder_name, 'datasets', img_folder_name])
    create_local_copy(base_folder_name)
    create_dataset(img_folder_name, subdir_l, augment=True, recursive=True)
    split_dataset()
