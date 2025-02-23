import os
import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)

import tensorflow as tf

from core.utilities.helper import get_files, get_directories, apply_rgb_mask, load_enumerations
from neural_network_playground.models.classification import ClassificationModel
from neural_network_playground.models.crop import CropModel


if getattr(sys, 'frozen', False):
    base_path = Path(sys.executable).resolve().parent
else:
    base_path = Path(__file__).resolve().parent


def confirm_exit():
    os.system('pause')
    sys.exit(1)

def load_config():
    """
    Load training hyperparameters from train_configuration.txt located in the same directory as this script.
    Expected keys (for example):
        crop_shape = (128, 128)
        classification_shape = (512, 512)
        validation_split = 0.2
        batch_size = 1
        lr = 1e-5
        crop_epochs = 25
        classification_epochs = 100
        seed = 42
    """
    config_file = base_path / "train_configuration.txt"
    print(f"Looking for config file at: {config_file}")
    if not config_file.is_file():
        print(
            "Configuration file train_configuration.txt not found in the current directory. Please select location of augmentation_config.txt")
        # Show file selection dialog
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        selected_file, _ = QFileDialog.getOpenFileName(None, "Select train configuration text file", str(base_path),
                                                       "Text Files (*.txt)")
        if not selected_file:
            print("No configuration file selected. Exiting.")
            confirm_exit()
        config_file = Path(selected_file)
    print(f"Using config file:{config_file}")

    config_data = {}
    with config_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"Invalid configuration line: {line}")
                confirm_exit()
            key, val = line.split("=", 1)
            config_data[key.strip()] = val.strip()

    config_schema = {
        "CROP_SHAPE": {"type": tuple, "length": 2, "default": (128, 128)},
        "CLASSIFICATION_SHAPE": {"type": tuple, "length": 2, "default": (512, 512)},
        "SEED": {"type": int, "default": 42},
    }

    config = {}
    for key, schema in config_schema.items():
        if key not in config_data:
            print(f"Missing configuration key: {key}")
            confirm_exit()
        try:
            value = eval(config_data[key], {"__builtins__": {}})
        except Exception as e:
            print(f"Error evaluating configuration key '{key}': {e}")
            confirm_exit()
        if schema["type"] == int:
            if not isinstance(value, int):
                print(f"Configuration key '{key}' must be an integer. Got {value} of type {type(value)}")
                confirm_exit()
        elif schema["type"] == tuple:
            if not isinstance(value, tuple) or len(value) != schema["length"]:
                print(f"Configuration key '{key}' must be a tuple of length {schema['length']}. Got {value}")
                confirm_exit()
            for element in value:
                if not isinstance(element, (int, float)):
                    print(
                        f"Configuration key '{key}' must be a tuple of numbers. Got element {element} of type {type(element)}")
                    confirm_exit()
        config[key] = value
    return config


def dir_dialog():
    app = QApplication.instance() or QApplication(sys.argv)
    selected_directory = QFileDialog.getExistingDirectory(None, "Select image catalog directory")
    if not selected_directory:
        print("No image catalog directory selected. Exiting.")
        confirm_exit()
    print(f"Using image catalog directory: {selected_directory}")
    return Path(selected_directory)


if __name__ == "__main__":
    # Ask user for the testrun directory (which should contain both models)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = load_config()
    # Load hyperparameter configuration from train_configuration.txt.
    crop_shape = config.get("crop_shape", (128, 128))
    classification_shape = config.get("classification_shape", (512, 512))
    seed = config.get("seed", 42)

    # Select trained model directory
    app = QApplication.instance() or QApplication(sys.argv)
    selected_directory = QFileDialog.getExistingDirectory(None, "Select trained model directory")
    if not selected_directory:
        print("No trained model directory selected. Exiting.")
        confirm_exit()
    print(f"Using trained model directory: {selected_directory}")
    testrun_dir = Path(selected_directory)

    # Select image catalog directory
    selected_directory = QFileDialog.getExistingDirectory(None, "Select augmented image catalog directory")
    if not selected_directory:
        print("No augmented image catalog directory selected. Exiting.")
        confirm_exit()
    print(f"Using augmented image catalog directory: {selected_directory}")
    catalog_path = Path(selected_directory)

    # Derive the testrun name from the provided directory.
    testrun_name = testrun_dir.name
    # Models are expected to be stored in subdirectories with names following this structure:
    crop_model_name = f"{testrun_name}_crop_model"
    classification_model_name = f"{testrun_name}_classification_model"

    # Instantiate the model classes.
    crop_model = CropModel(crop_shape)
    classification_model = ClassificationModel(classification_shape)

    # Build a list of test sample paths.
    test_samples = []
    # We assume that the catalog path contains a subdirectory "images"
    images_dir = catalog_path / "images"
    if not images_dir.is_dir():
        print(f"'images' directory not found under the provided catalog path: {images_dir}")
        confirm_exit()

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
        confirm_exit()

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
    plt.tight_layout()
    plt.savefig(f"{testrun_name}_confusion_matrix.png", bbox_inches='tight')
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
    confirm_exit()
