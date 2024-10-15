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


def test_set_classes(rp, test_dir, verbose=False):
    """
    This function is used to compute the accuracy of a model using a test directory which contains images and
    the corresponding annotations. For each image the pipeline returns a set of classes, the prediction is
    correct if the predicted classes set contains the true classes of the image written in the annotation.

    :param rp: The instance of the recognition pipeline to use.
    :param test_dir: The directory to test, which contains images and annotations.
    :param verbose: If true it prints more information about the recognition.
    """
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']

    errors = []
    tot_imgs = 0
    correct = 0
    proposed=0
    for image_path in os.listdir(test_dir):

        full_path = os.path.join(test_dir, image_path)
        name_ext = os.path.splitext(full_path)
        true_classes = set()

        if name_ext[1].lower() in valid_image_extensions:

            print(f"\n\nExtracting annotation: {name_ext[0]}.txt")
            with open(name_ext[0]+'.txt', 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    true_classes.add(rp.classes_dict[class_id])

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
            else:
                errors.append(full_path)

            if verbose:
                print("- True classes: ", end=" ")
                for c in true_classes:
                    print(c, end=" ")

                print("\n- Predicted classes: ", end=" ")
                for c in predicted_classes_name:
                    print(c, end=" ")
        #else:
        #    print(f"\nSkipping non-image file: {full_path}")


    print('\n\n---------------------------------------------')
    print(f'Total images: {tot_imgs}')
    print(f'Correct predictions: {correct}')
    print('---------------------------------------------')
    print(f'Total number of classes: {len(rp.classes_dict)}')
    print(f'Average number of proposed predictions: {proposed/tot_imgs}')
    print('---------------------------------------------')
    print('Images with an error in the recognition:')
    for image_path in errors:
        print(image_path)


if __name__ == '__main__':

    yolov5_dir = '/home/christofer/PycharmProjects/computerVision/yolov5'
    weights_path = '/home/christofer/PycharmProjects/computerVision/yolov5/runs/train/10C_40E_DA/weights/best.pt'
    test_path = '/home/christofer/Desktop/test'

    rp = RecognitionPipeline(yolov5_dir=yolov5_dir, custom_weights=weights_path)
    test_set_classes(rp, test_path, verbose=True)