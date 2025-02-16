import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)

# from core.utilities.helper import get_directories, get_files
# from kunstkammer.neural_network_playground.classification import ClassificationModel
# from kunstkammer.neural_network_playground.core.helper import apply_rgb_mask, load_enumerations
# from kunstkammer.neural_network_playground.crop import CropModel
import tensorflow as tf

from image_collector.core.utilities.helper import get_files, get_directories, apply_rgb_mask
from neural_network_playground.core.helper import load_enumerations
from neural_network_playground.models.classification import ClassificationModel
from neural_network_playground.models.crop import CropModel

trained_model_dir = Path(os.path.dirname(__file__), "trained")
# catalog_path = Path("D:/Projects/bachelor_thesis/NN-Coin-Recognizer/ImageCollector/coin_catalog/augmented_50")
crop_shape = (128, 128)
classification_shape = (512, 512)

# crop_model_name = "image_dataset_first_testrun_crop_model"
# classification_model_name = "first_testrun_classification_model_30"

if __name__ == "__main__":
    # Ask user for the testrun directory (which should contain both models)
    testrun_dir_input = input("Enter the testrun directory (containing crop and classification models): ").strip()
    testrun_dir = Path(testrun_dir_input)
    if not testrun_dir.is_dir():
        print(f"Provided testrun directory does not exist: {testrun_dir}")
        exit(-1)

    # Derive the testrun name from the provided directory.
    testrun_name = testrun_dir.name
    # Models are expected to be stored in subdirectories with names following this structure:
    crop_model_name = f"{testrun_name}_crop_model"
    classification_model_name = f"{testrun_name}_classification_model"

    # Ask user for the catalog path (the directory that contains the images folder)
    catalog_path_input = input("Enter the catalog path: ").strip()
    catalog_path = Path(catalog_path_input)
    if not catalog_path.is_dir():
        print(f"Provided catalog path does not exist: {catalog_path}")
        exit(-1)

    # Instantiate the model classes.
    crop_model = CropModel(crop_shape)
    classification_model = ClassificationModel(classification_shape)

    # Build a list of test sample paths.
    test_samples = []
    # We assume that the catalog path contains a subdirectory "images"
    images_dir = catalog_path / "images"
    if not images_dir.is_dir():
        print(f"'images' directory not found under the provided catalog path: {images_dir}")
        exit(-1)

    for coin_dir_path in get_directories(images_dir):
        for image_path in get_files(coin_dir_path):
            # Each sample is stored as "coin_category/filename"
            sample = f"{image_path.parts[-2]}/{image_path.parts[-1]}"
            test_samples.append(sample)

    # Load the models and the enumeration (class names)
    try:
        crop_model_dir = testrun_dir / crop_model_name
        classification_model_dir = testrun_dir / classification_model_name

        crop_model.load_model(crop_model_dir)
        classification_model.load_model(classification_model_dir)
        enumerations = load_enumerations(classification_model_dir)
        print("=== Model Data loaded ===")
    except Exception as ex:
        print("=== Could not load models/datasets ===")
        print(f"Exception: {ex}")
        exit(-1)

    true_classes = []
    predicted_classes = []
    results = {"pass": [], "fail": []}

    # Process each test sample.
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
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Compute and print additional metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Optionally, print a full classification report
    report = classification_report(true_classes, predicted_classes, labels=labels, zero_division=0)
    print("\nClassification Report:")
    print(report)
