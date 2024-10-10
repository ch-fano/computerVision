from cross_validation import create_k_fold
from pathlib import Path
import datetime
import os

def check_matches(dir_path):
    images = []
    for img in os.listdir(dir_path / "images"):
        images.append(img)

    labels = []
    for label in os.listdir(dir_path / "labels"):
        labels.append(label)

    removed_imgs = []
    for img in images:
        img_name = os.path.splitext(img)[0]

        if img_name + ".txt" in labels:
            labels.remove(img_name + ".txt")
            removed_imgs.append(img)

    for img in removed_imgs:
        images.remove(img)

    print(f"\tImages: {images}")
    print(f"\tLabels: {labels}")




def test_k_fold(dataset_path, yaml_file, ksplit):
    create_k_fold(dataset_path, yaml_file, ksplit)

    base_path = Path(Path(dataset_path) / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")

    for k in range(1, ksplit+1):
        print(f"\nSplit {k}")

        print("- Train: ")
        train_split_path = base_path / f"split_{k}" / "train"
        check_matches(train_split_path)

        print("- Validation: ")
        val_split_path = base_path / f"split_{k}" / "val"
        check_matches(val_split_path)


if __name__ == '__main__':

    dataset_path = "/home/christofer/PycharmProjects/computerVision/datasets"
    yaml_file = "/yolov5/data/comunichiamo_10_classes.yaml"
    ksplit = 5

    test_k_fold(dataset_path, yaml_file, ksplit)