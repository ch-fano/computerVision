import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_confusion_matrix(data, x_classes, y_classes):
    """
        This function creates the images of the confusion matrix, using the information passed.

        :param data: The values to visualize in the confusion matrix
        :param x_classes: The classes to visualize as the column names.
        :param y_classes: The classes to visualize as the row names.
    """

    # Create a dataframe
    df_cm = pd.DataFrame(data, index=y_classes, columns=x_classes)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.ylabel("Predicted")
    plt.xlabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    # Confusion matrix data with background class
    data = np.array([[36, 7, 0],
                     [2, 56, 1],
                     [1, 2, 10],
                     [0, 6, 46],
                     [9, 24, 2]])

    tot_elem = np.array([48, 95, 59])

    # Divide each element in 'data' by the corresponding element in 'tot_elem' for each column
    normalized_matrix = data / tot_elem

    # Define class names including background
    x_classes = ['Buca', 'Rifiuti', 'Illuminazione']
    y_classes = ["Buca", "Rifiuti", "Illuminazione (yolo)", "Illuminazione (luminosità)", "Background"]

    make_confusion_matrix(normalized_matrix, x_classes, y_classes)