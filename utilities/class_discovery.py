import os
from recognition_pipeline import RecognitionPipeline

def test_best_class(rp, directory_path):
    """
    This function recognise the objects in the images of the directory, memorize and print only the results
    of the class with the highest probability.

    :param rp: The instance of the recognition pipeline to use.
    :param directory_path: The path to the directory to test.
    """
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
    dir_classes = {'undefined': 0}
    tot_imgs = 0

    for image_path in os.listdir(directory_path):

        full_path = os.path.join(directory_path, image_path)

        if os.path.splitext(full_path)[1].lower() in valid_image_extensions:
            print(f"\nProcessing image: {full_path}")
            tot_imgs += 1

            classes = rp.recognize(full_path)
            best = classes[0]

            if len(best) == 0:
                dir_classes['undefined'] += 1
            elif best[0] in dir_classes:
                dir_classes[best[0]] += 1
            else:
                dir_classes[best[0]] = 1
        else:
            print(f"Skipping non-image file: {full_path}")

    print(f'Total images: {tot_imgs}')
    print('Predicted results: ')
    for key, value in dir_classes.items():
        print(f'- {key}: {value}')


def test_set_classes(rp, directory_path, classes_txt, verbose=False):
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']

    class_labels = list()
    with open(classes_txt, 'r') as f:
        for line in f:
            class_labels.append(line.strip())

    tot_imgs = 0
    correct = 0
    proposed=0
    for image_path in os.listdir(directory_path):

        full_path = os.path.join(directory_path, image_path)
        name_ext = os.path.splitext(full_path)
        true_classes = set()

        if name_ext[1].lower() in valid_image_extensions:

            print(f"\n\nExtracting annotation: {name_ext[0]}.txt")
            with open(name_ext[0]+'.txt', 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    true_classes.add(class_labels[class_id])

            print(f"Processing image: {full_path}")
            tot_imgs += 1

            predicted_classes = rp.recognize(full_path)
            predicted_classes_name =  set([elem[0] for elem in predicted_classes])
            proposed += len(predicted_classes_name)

            # Calculate if it is correct or not
            recognized = True
            for c in true_classes:
                if c not in predicted_classes_name:
                    recognized = False

            if recognized:
                correct += 1

            if verbose:
                print("- True classes: ", end=" ")
                for c in true_classes:
                    print(c, end=" ")

                print("\n- Predicted classes: ", end=" ")
                for c in predicted_classes_name:
                    print(c, end=" ")
        #else:
        #    print(f"\nSkipping non-image file: {full_path}")

    print(f'\n\nTotal images: {tot_imgs}')
    print(f'Correct predictions: {correct}')
    print(f'Total number of classes: {len(class_labels)}')
    print(f'Average number of proposed predictions: {proposed/tot_imgs}')


if __name__ == '__main__':

    yolov5_dir = '/home/christofer/PycharmProjects/computerVision/yolov5'
    weights_path = '/home/christofer/PycharmProjects/computerVision/yolov5/runs/train/10C_40E_DA_3_2_MAX/weights/best.pt'
    rp = RecognitionPipeline(yolov5_dir=yolov5_dir, custom_weights=weights_path)

    dir_path = '/home/christofer/Desktop/test'
    class_file = '/home/christofer/PycharmProjects/computerVision/classes.txt'
    test_set_classes(rp, dir_path, class_file, verbose=True)