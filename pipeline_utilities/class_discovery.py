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

            if len(classes) == 0:
                dir_classes['undefined'] += 1
            else:
                best_class = classes[0][0]     # the first index extract the best, the second extract the class name
                if best_class in dir_classes:
                    dir_classes[best_class] += 1
                else:
                    dir_classes[best_class] = 1
        else:
            print(f"Skipping non-image file: {full_path}")

    print(f'Total images: {tot_imgs}')
    print('Predicted results: ')
    for key, value in dir_classes.items():
        print(f'- {key}: {value}')

def extract_true_classes(img_path, classes_dict, extraction_method):
    """
    This function extracts the true classes of the images provided according to the specified extraction method.

    :param img_path: The path to the image.
    :param classes_dict: The dictionary which contains the { ID : class_name } association.
    :param extraction_method: The method to use for the extraction of the true classes:
        - NAME: each image has name which starts with id___, id is used as key of the db_ids dictionary to define the
                true class of the image.
        - ANNOTATION: search for the annotation img_name.txt and extracts the true classes from it.
    :return: A set of the classes name for which is present an instance in the image.
    """

    true_classes = set()
    if extraction_method == 'NAME':

        db_ids = {
            1: 'buca',
            4: 'semaforo_non_funzionante',
            11: 'segnaletica_danneggiata',
            14: 'graffiti',
            20: 'veicolo_abbandonato',
            21: 'bicicletta_abbandonata',
            22: 'illuminazione',
            27: 'deiezioni_canine',
            156: 'siringa_abbandonata',
            159: 'rifiuti',
        }

        img_name = os.path.basename(img_path)
        print(f"\n\nExtracting class id from the image name: {img_name}")
        db_class_id = int(img_name[:img_name.find('_')])     # each image has name which starts with id___
        true_classes.add(db_ids[db_class_id])

    elif extraction_method == 'ANNOTATION':

        img_name_path = os.path.splitext(img_path)[0]
        print(f"\n\nExtracting annotation: {img_name_path}.txt")
        with open(img_name_path + '.txt', 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                true_classes.add(classes_dict[class_id])

    else:
        raise ValueError(f"Invalid true classes extracting method: {extraction_method}")

    return true_classes

def test_set_classes(rp, test_dir, extracting_true_classes='NAME', verbose=False):
    """
    This function is used to compute the accuracy of a model using a test directory which contains images and
    the corresponding annotations. For each image the pipeline returns a set of classes, the prediction is
    correct if the predicted classes set contains the true classes of the image written in the annotation.

    :param rp: The instance of the recognition pipeline to use.
    :param test_dir: The directory to test, which contains images and annotations.
    :param verbose: If True it prints more information about the recognition.
    """
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']

    extracting_methods = ['NAME', 'ANNOTATION']
    if extracting_true_classes not in extracting_methods:
        raise ValueError(f"Invalid true classes extracting method: {extracting_true_classes}\n"
                         f"The allowed values are: {extracting_methods}")

    errors = []
    tot_imgs = 0
    correct = 0
    proposed=0
    prediction_distribution = {}

    for image_path in os.listdir(test_dir):

        full_path = os.path.join(test_dir, image_path)
        extension = os.path.splitext(full_path)[1]

        if extension.lower() in valid_image_extensions:

            true_classes = extract_true_classes(full_path, rp.classes_dict, extracting_true_classes)

            print(f"Processing image: {full_path}")
            tot_imgs += 1

            predicted_classes = rp.recognize(full_path)
            predicted_classes_name =  set([elem[0] for elem in predicted_classes])
            proposed += len(predicted_classes_name)
            prediction_distribution[len(predicted_classes_name)] = (
                    prediction_distribution.get(len(predicted_classes_name), 0) + 1)

            # Calculate if it is correct or not
            recognized = True
            for c in true_classes:
                if c not in predicted_classes_name:
                    recognized = False

            if recognized:
                correct += 1
            else:
                errors.append({
                    'name' : full_path,
                    'true' : true_classes,
                    'predicted' : predicted_classes_name,
                })

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
    print('Prediction distribution (proposed classes - number):')
    for key, value in sorted(prediction_distribution.items()):
        print(f'-{key}: {value}')
    print('---------------------------------------------')
    print('Images with an error in the recognition (name: true - predicted):')
    for error_dict in errors:
        print(f"{error_dict['name']}: {error_dict['true']} - {error_dict['predicted']}")

if __name__ == '__main__':

    yolov5_dir = '/home/christofer/PycharmProjects/computerVision/yolov5'
    weights_path = '/home/christofer/PycharmProjects/computerVision/yolov5/runs/train/FIN_fold_4/weights/best.pt'
    test_path = '/home/christofer/Desktop/test'


    rp = RecognitionPipeline(yolov5_dir=yolov5_dir, custom_weights=weights_path)
    rp.set_common_classes(["buca", "rifiuti"])
    test_set_classes(rp, test_path, extracting_true_classes='ANNOTATION', verbose=True)
    #test_best_class(rp, test_path)