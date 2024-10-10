import albumentations as A
import cv2
import os


def data_augmentation(img_path, label_path):
    """
    This function applies data augmentation to the specified image preserving the label correctness.

    :param img_path: The path to the image to augment.
    :param label_path: The path to the label of the image to augment.
    :return:
    """

    data_augmentation.counter = getattr(data_augmentation, 'counter', 0) + 1

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        #A.Equalize(always_apply=True, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1, always_apply=True),
        A.Rotate(limit=(-10, 10), p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append((int(class_id), x_center, y_center, width, height))

    bbox_coordinates = [bbox[1:] for bbox in bboxes]
    category_ids = [bbox[0] for bbox in bboxes]

    transformed = transform(image=image, bboxes=bbox_coordinates, category_ids=category_ids)

    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_ids = transformed['category_ids']

    transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

    img_root_ext = os.path.splitext(img_path)
    label_root_ext = os.path.splitext(label_path)

    new_img_path = img_root_ext[0] + f'_{data_augmentation.counter}' + img_root_ext[1]
    new_label_path = label_root_ext[0] + f'_{data_augmentation.counter}' + label_root_ext[1]

    cv2.imwrite(new_img_path, transformed_image_bgr)

    with open(new_label_path, 'w') as f:
        i = 0
        for bbox in transformed_bboxes:
            class_id = transformed_ids[i]
            x_center, y_center, width, height = bbox
            f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

            i += 1

    return new_img_path
