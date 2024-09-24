import os
import shutil
import random
from sklearn.model_selection import train_test_split
from utilities.data_augmentation import data_augmentation



def create_local_copy(src_path, copy_dir_name):
    """
    This function creates a local copy the specified directory in the project directory.

    :param src_path: The directory to copy.
    :param copy_dir_name: The name of the folder to copy the image to.
    """
    if os.path.isdir(src_path):
        shutil.copytree(src_path, os.path.join(os.getcwd(), copy_dir_name))
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

def create_dataset_folders(dataset_name):
    """
    This function creates the dataset folders of the images and labels.
    :param dataset_name: The name of the dataset to create.
    """

    # Create the datasets directory
    parent_dir_datasets = os.path.join(os.getcwd(), dataset_name)
    os.mkdir(parent_dir_datasets)

    parent_dir_images = os.path.join(os.getcwd(), dataset_name, 'images')
    parent_dir_labels = os.path.join(os.getcwd(), dataset_name, 'labels')

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
            shutil.rmtree(os.path.join(os.getcwd(), d), ignore_errors=False)

    print('---- Removed old folders ----')


def split_dataset(tmp_folder_path, dataset_name, augment=False, recursive=False):
    """
    This function randomly splits the dataset of images and labels into train, test and validation sets.

    :param tmp_folder_path: The temporary folder which contains the images to  split.
    :param dataset_name: The name of the directory which will contain the splits.
    :param augment: If 'True' it applies the data augmentation on the train set.
    :param recursive: If 'True' it applies the data augmentation also on the images already augmented.
    """

    # Read images and annotations
    images = []
    annotations = []
    for f in os.listdir(tmp_folder_path):
        if f[-3:] == 'txt':
            annotations.append(os.path.join(tmp_folder_path, f))
        else:
            image_path = os.path.join(tmp_folder_path, f)

            if os.path.exists(os.path.splitext(image_path)[0]+'.txt'):
                images.append(image_path)
            else:
                print(f'No annotation for the image: {image_path}')

    images.sort()
    annotations.sort()

    # Split the dataset into train-valid-test splits
    random_seed = random.randint(0, 10000)

    # train_test_split apply a synchronous shuffle to the images and annotations to preserve the relation between them
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations,
                                                                                    test_size=0.2,
                                                                                    random_state=random_seed)

    # divides the previous validation set into test and validation
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                                  test_size=0.5,
                                                                                  random_state=random_seed)
    create_dataset_folders(dataset_name)

    # Move the splits into their folders
    move_files_to_folder(train_images, os.path.join(os.getcwd(), dataset_name, 'images', 'train'))
    move_files_to_folder(val_images, os.path.join(os.getcwd(), dataset_name, 'images', 'val'))
    move_files_to_folder(test_images, os.path.join(os.getcwd(), dataset_name, 'images', 'test'))
    move_files_to_folder(train_annotations, os.path.join(os.getcwd(), dataset_name, 'labels', 'train'))
    move_files_to_folder(val_annotations, os.path.join(os.getcwd(), dataset_name, 'labels', 'val'))
    move_files_to_folder(test_annotations, os.path.join(os.getcwd(), dataset_name, 'labels', 'test'))

    remove_dataset_folders([tmp_folder_path])
    print('---- Dataset successfully splitted ----')

    if augment:
        apply_data_augmentation(os.path.join(os.getcwd(), dataset_name), recursive)


def apply_data_augmentation(dir_name, recursive=False):
    """
    This function apply data augmentation to the train set of the specified directory

    :param dir_name: The directory to apply data augmentation.
    :param recursive: If 'True' it applies the data augmentation also on the images already augmented .
    """

    print('---- Starting the data augmentation ----')
    images_dir = os.path.join(dir_name, 'images', 'train')
    labels_dir = os.path.join(dir_name, 'labels', 'train')

    class_instances = {}
    class_images = {}
    for img_name in os.listdir(images_dir):
        class_id = img_name[:img_name.find('_')]     # each image has name which starts with id___

        # count the number of elements
        class_instances[class_id] = class_instances.get(class_id, 0) + 1

        if class_id not in class_images:
            class_images[class_id] = []
        class_images[class_id].append(img_name)


    augmented_images = 0
    tot_instances = max(class_instances.values()) + min(class_instances.values())


    for class_id, images in class_images.items():

        # Create new images to have the same number of pictures for each category
        while class_instances[class_id] < tot_instances:
            augmented_images += 1
            img_name = random.choice(images)
            label_name = os.path.splitext(img_name)[0] + '.txt'

            new_img_path = data_augmentation(
                os.path.join(images_dir, img_name),
                os.path.join(labels_dir, label_name)
            )

            class_instances[class_id] += 1
            if recursive:
                images.append(os.path.basename(new_img_path))

    print(f'---- Finished data augmentation: augmented {augmented_images} images ----')


def create_dataset(dir_path, subdir_list, tmp_folder_name, delete_dir=False):
    """
    This function moves all the images and labels in the specified subdirectories in a unique folder.

    :param dir_path: The path to the common base directory.
    :param subdir_list: The subdirectories with the images and labels to move.
    :param tmp_folder_name: The name of the temporary folder on which the images have to be moved.
    :param delete_dir: If 'True' it deletes the folder passed as dir_path.
    """

    if os.path.exists(os.path.join(os.getcwd(), 'classes.txt')):
        os.remove(os.path.join(os.getcwd(), 'classes.txt'))

    shutil.move(os.path.join(dir_path, 'classes.txt'), os.getcwd())

    files = []
    # For each directory extract the files and create a list of them
    for subdir in subdir_list:
        subdir_path = os.path.join(dir_path, subdir)
        files += [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f != 'classes.txt']

    # Create the new temporary folder and move files on it
    os.mkdir(os.path.join(os.getcwd(), tmp_folder_name))
    move_files_to_folder(files, tmp_folder_name)

    if delete_dir:
        remove_dataset_folders([dir_path])

    print('---- Created the dataset ----')

def setup_dataset(base_dir, subdir_list, dataset_name='datasets', temp_dir='temp_folder', augment=False, recursive=False):
    """
    This functions setups the dataset for the training of the model, you can

    :param base_dir:
            The root directory that contains the image dataset. This directory should be structured with
            subdirectories, where each subdirectory corresponds to a specific class. The images for each
            class are stored within their respective subdirectory. The directory structure should follow
            the format:

            base_dir/
            ├── class_1/
            │   ├── img_1.jpg
            │   ├── img_2.jpg
            │   └── ...
            ├── class_2/
            │   ├── img_1.jpg
            │   ├── img_2.jpg
            │   └── ...
            └── ...
    :param subdir_list: The list of subdirectories containing the images and labels to use during the training.
    :param dataset_name: The name to assign to the dataset directory which will be created.
    :param temp_dir: The name of the temporary folder that will serve as a local copy of the `base_dir`. This folder
                     is created to work on the images without modifying the original `base_dir`.
    :param augment: If 'True' it applies the data augmentation on the train set.
    :param recursive: If 'True' it applies the data augmentation also on the images already augmented.
    """

    base_dir_copy = os.path.basename(base_dir) + '_copy'

    # Remove previous folders with the same name
    remove_dataset_folders([temp_dir, dataset_name, base_dir_copy])

    # Creates a local copy of the base directory
    create_local_copy(base_dir, base_dir_copy)

    # Extract all the images in the subdirectories list and moves them in a temporary folder
    create_dataset(os.path.join(os.getcwd(), base_dir_copy), subdir_list, temp_dir, delete_dir=True)

    # Divide the dataset into train, test and validation
    split_dataset(os.path.join(os.getcwd(), temp_dir), dataset_name, augment, recursive)

if __name__ == '__main__':
    #base_img_folder_path = '/home/christofer/Desktop/cv_images'  # Your path to the photo
    #subdir_l = ['1_strada_buca',
    #            '4_semaforo_non_funzionante',
    #            '11_segnaletica_danneggiata',
    #            '14_graffiti',
    #            '20_veicolo_abbandonato',
    #            '21_bicicletta_abbandonata',
    #            '22_strada_al_buio',
    #            '27_deiezioni_canine',
    #            '156_siringa_abbandonata',
    #            '159_rifiuti_abbandonati']  # Add your subdir

    base_img_folder_path = '/home/christofer/Desktop/first_images/'
    subdir_l = ['1_strada_buca', '22_strada_al_buio', '159_rifiuti_abbandonati']

    setup_dataset(base_img_folder_path, subdir_l)