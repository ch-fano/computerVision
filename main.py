import os
import shutil
import random
from sklearn.model_selection import train_test_split
from data_augmentation import data_augmentation as DA

# Define the folder name to personalize the script
img_folder_name = 'img_copy'
tmp_folder_name = 'tmp_folder'
base_folder_name = ''  # your path to the photo
subdir_l = []  # add your subdir


def create_local_copy(src_path):
    if os.path.isdir(src_path):
        shutil.copytree(src_path, os.path.join(os.getcwd(), img_folder_name))
        print('---- Created local copy ----')


# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


def create_dataset_folders():
    # Create the datasets directory
    parent_dir_datasets = os.path.join(os.getcwd(), 'datasets')
    os.mkdir(parent_dir_datasets)

    parent_dir_images = os.path.join(os.getcwd(), 'datasets/images')
    parent_dir_labels = os.path.join(os.getcwd(), 'datasets/labels')

    # Create the parent directories
    os.mkdir(parent_dir_images)
    os.mkdir(parent_dir_labels)

    # Create che the subdirectories
    for new_dir in ['train', 'val', 'test']:
        # Create directory for images
        path = os.path.join(parent_dir_images, new_dir)
        os.mkdir(path)

        # Create directory for labels
        path = os.path.join(parent_dir_labels, new_dir)
        os.mkdir(path)

    print('---- Created images and labels directories ----')


def remove_dataset_folders(dir_list):
    for d in dir_list:
        if os.path.isdir(d):
            shutil.rmtree(os.path.join(os.getcwd(), d), ignore_errors=False, onerror=None)

    print('---- Removed old folders ----')


def split_dataset():
    # Read images and annotations
    images = []
    annotations = []
    for f in os.listdir(tmp_folder_name):
        if f[-3:] == 'txt':
            annotations.append(os.path.join(tmp_folder_name, f))
        else:
            images.append(os.path.join(tmp_folder_name, f))

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
    move_files_to_folder(train_images, 'datasets/images/train')
    move_files_to_folder(val_images, 'datasets/images/val/')
    move_files_to_folder(test_images, 'datasets/images/test/')
    move_files_to_folder(train_annotations, 'datasets/labels/train/')
    move_files_to_folder(val_annotations, 'datasets/labels/val/')
    move_files_to_folder(test_annotations, 'datasets/labels/test/')

    remove_dataset_folders([tmp_folder_name, img_folder_name])
    print('---- Dataset successfully splitted ----')


def data_augmentation(dir_name, subdir_list, recursive=False):

    tot_files = []
    for subdir in subdir_list:
        path = os.path.join(dir_name, subdir)
        tot_files.append(len([name for name in os.listdir(path)]) // 2)

    i = 0
    max_num = max(tot_files) + max(tot_files)//2

    # extract the images from the subdirectory
    for subdir in subdir_list:
        path = os.path.join(dir_name, subdir)
        imgs = [os.path.join(path, f) for f in os.listdir(path) if f[-3:] != 'txt']

        # create new images to have the same number of pictures for each category
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

    if augment:
        data_augmentation(dir_name, subdir_list, recursive)

    # TO DO: understand where to put classes.txt
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
    remove_dataset_folders([tmp_folder_name, 'datasets', img_folder_name])
    create_local_copy(base_folder_name)
    create_dataset(img_folder_name, subdir_l, augment=True, recursive=True)
    split_dataset()
