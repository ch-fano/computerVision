from sklearn.model_selection import KFold
from collections import Counter
from pathlib import Path
import pandas as pd
import datetime
import shutil
import yaml
import sys
import pickle

def create_k_fold(dataset_path, yaml_file, ksplit):
    print("Starting the splitting of the dataset...")

    dataset_path = Path(dataset_path)
    labels = sorted((dataset_path / "labels").rglob("*.txt"))

    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())

    indx = [label.stem for label in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    for label in labels:
        lbl_counter = Counter()

        with open(label, "r") as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ")[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    pd.set_option('future.no_silent_downcasting', True)
    labels_df = labels_df.fillna(0.0).infer_objects(copy=False)

    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)

    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.iloc[train].index, f"split_{idx}"] = "train"
        folds_df.loc[labels_df.iloc[val].index, f"split_{idx}"] = "val"

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    supported_extensions = [".jpg", ".jpeg", ".png"]

    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    # Sort the images to match the label order
    images.sort()

    # Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    for image, label in zip(images, labels):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

    # opzionale
    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

    print("Finished the splitting of the dataset")
    return ds_yamls

def cross_validation(yolov5_path, ksplit, ds_yamls, starting_split=1):
    print("Starting the cross-validation...")

    # Add the directory of the file to sys.
    sys.path.insert(0, str(Path(yolov5_path)))

    import train, val

    weights_path = yolov5_path + "/yolov5s.pt"
    results = {}

    # Define your additional arguments here
    batch = 8
    project = yolov5_path + "/runs/train"
    epochs = 40

    for k in range(starting_split, ksplit+1):
        print(f"Evaluating split {k}/{ksplit}")

        dataset_yaml = ds_yamls[k]

        # Train the model for the current fold
        train.run(data=dataset_yaml,
                  weights=weights_path,
                  epochs=epochs,
                  batch_size=batch,
                  project=project,
                  name=f'fold_{k}')

        # Run validation to get metrics
        metrics = val.run(data=dataset_yaml, weights=f'{yolov5_path}/runs/train/fold_{k}/weights/best.pt')

        results[k] = metrics  # save output metrics for further analysis

        with open(f"result_fold{k}.pickle", "wb") as result_file:
            pickle.dump(metrics, result_file)

    print("Finished the cross-validation")
    print(results)

    with open("metrics.pickle", "wb") as metrics_file:
        pickle.dump(results, metrics_file)

def extract_yamls(kfold_path, ksplit):
    print("The dataset already exists, skipping creation")
    print("Extracting yamls...")
    ds_yamls = []

    for k in range(1, ksplit+1):
        ds_yamls.append(kfold_path / f"split_{k}/split_{k}_dataset.yaml")

    print("Finished extracting yamls")
    return ds_yamls

def apply_cross_validation(dataset_path, yaml_file, yolov5_path, ksplit, starting_split=1):
    kfold_path = Path(Path(dataset_path) / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")

    if kfold_path.exists():
        dataset_yamls = extract_yamls(kfold_path, ksplit)
    else:
        dataset_yamls = create_k_fold(dataset_path, yaml_file, ksplit)

    cross_validation(yolov5_path, ksplit, dataset_yamls, starting_split)


if __name__ == "__main__":
    ksplit = 5

    apply_cross_validation(
        "/home/christofer/PycharmProjects/computerVision/datasets",
        "/home/christofer/PycharmProjects/computerVision/yolov5/data/comunichiamo.yaml",
        "/home/christofer/PycharmProjects/computerVision/yolov5",
        ksplit
    )