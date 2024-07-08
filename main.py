import os
import shutil
from sklearn.model_selection import train_test_split

# Define the folder name to personalize the script
img_folder_name = 'img_copy'
tmp_folder_name = 'tmp_folder'
base_folder_name = '/home/christofer/Desktop/photos'

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
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2,
                                                                                    random_state=1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                                  test_size=0.5, random_state=1)
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


def create_dataset(dir_name, subdir_list):

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
    create_dataset(img_folder_name, ['1', '22', '159'])
    split_dataset()
