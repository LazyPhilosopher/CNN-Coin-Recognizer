import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from core.utilities.helper import get_directories, get_files
from kunstkammer.neural_network_playground.classification import ClassificationModel
from kunstkammer.neural_network_playground.core.helper import apply_rgb_mask, load_enumerations
from kunstkammer.neural_network_playground.crop import CropModel
import tensorflow as tf


trained_model_dir = Path(os.path.dirname(__file__), "trained")
catalog_path = Path("D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/coin_catalog/augmented_50")
crop_shape = (128, 128)
classification_shape = (512, 512)

crop_model_name = "remastered_crop_model_50"
classification_model_name = "remastered_classification_model_200"

if __name__ == "__main__":

    crop_model = CropModel(crop_shape)
    classification_model = ClassificationModel(classification_shape)

    # test_samples = [
    #     "(Czech Republic, 50 Korun, 2008)/10_19.png",
    #     "(Czech Republic, 1 Koruna, 2018)/9_80.png",
    #     "(USA, 2.5 Dollar, 1909)/2_24.png",
    #     "(Great britain, 0.5 Souvereign, 1906)/3_3.png",
    #     "(Iran, 0.5 Souvereign, 1925)/4_3.png",
    #     "(Austria-Hungary, 20 Korona, 1893)/0_80.png",
    #     "(Czech Republic, 10 Korun, 2020)/0_31.png",
    #     "(France, 1 Franc, 1918)/4_21.png",
    #     "(Czech Republic, 5 Korun, 2002)/10_27.png",
    #     "(India, 1 Rupee, 1840)/8_26.png",
    #     "(Austria-Hungary, 20 Korona, 1893)/0_15.png",
    # ]

    test_samples = []
    for coin_dir_path in get_directories(Path(catalog_path, "images")):
        for image_path in get_files(coin_dir_path):
            test_samples.append(str(image_path.parts[-2] + "/"+ image_path.parts[-1]))

    try:
        crop_model.load_model(Path(trained_model_dir, crop_model_name))
        classification_model.load_model(Path(trained_model_dir, classification_model_name))
        enumerations = load_enumerations(Path(trained_model_dir, classification_model_name))
        print("=== Model Datas loaded ===")

    except Exception as ex:
        print(f"=== Could not load datasets ===")
        print(f"Exception: {ex}")
        exit(-1)

    # print("=== Model Summary ===")
    # print(crop_model.model.summary())
    # print(classification_model.model.summary())

    true_classes = []
    predicted_classes = []
    results = {"pass": [], "fail": []}
    for sample_path in test_samples:
        full_sample_path = str(Path(catalog_path, "images", sample_path))
        small_image = crop_model.load_image(full_sample_path)
        image_full = classification_model.load_image(full_sample_path)

        small_image_batch = tf.expand_dims(small_image, 0)
        bw_mask = crop_model.predict_mask(small_image_batch, threshold=0.003, verbose=0)
        bw_mask = tf.image.resize(bw_mask, classification_shape)

        masked_image = apply_rgb_mask(image_full, bw_mask)
        masked_image_batch = tf.expand_dims(masked_image, 0)

        predictions = classification_model.predict(masked_image_batch)
        result = tf.nn.softmax(predictions)

        true_class = Path(sample_path).parts[0]
        predict_class = enumerations[np.argmax(result)]

        # Append true and predicted classes to the lists
        true_classes.append(true_class)
        predicted_classes.append(predict_class)

        if true_class == predict_class:
            results["pass"].append(sample_path)
        else:
            results["fail"].append(sample_path)

        print(f"[{'PASS' if true_class == predict_class else 'FAIL'}] {true_class} -> {predict_class}")

    print(f"Total Pass/Fail: {len(results['pass'])}/{len(results['fail'])}")

    # Generate the confusion matrix
    labels = list(enumerations)  # Get the list of class names from the enumerations
    cm = confusion_matrix(true_classes, predicted_classes, labels=labels)

    # Display and/or save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

    # Optionally save the confusion matrix as an image
    plt.savefig("confusion_matrix.png")

    # Show the confusion matrix in a pop-up window
    plt.title("Confusion Matrix")
    plt.show()
