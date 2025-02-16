import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from neural_network_playground.core.helper import construct_pairs, create_dataset, get_directories
from neural_network_playground.core.models import build_resnet34_model, build_conv_model
from neural_network_playground.models.classification import ClassificationModel
from neural_network_playground.models.crop import CropModel


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
    # Use the directory of the current file
    config_file = Path(__file__).resolve().parent / "train_configuration.txt"
    if not config_file.is_file():
        print("Configuration file train_configuration.txt not found in the script directory!")
        sys.exit(1)

    config = {}
    with config_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"Invalid configuration line: {line}")
                sys.exit(1)
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                # Use eval() to allow arithmetic expressions (e.g. "1e-5" or "(128, 128)")
                config[key] = eval(value, {"__builtins__": {}})
            except Exception as e:
                print(f"Error evaluating configuration for key '{key}': {e}")
                sys.exit(1)
    return config


if __name__ == "__main__":
    # Ask the user for the input directory (dataset) and the testrun name.
    input_dir = input("Enter the input directory for training data: ").strip()
    testrun_name = input(
        "Enter the testrun name (this will be used for the output directory and model names): ").strip()

    # Set paths relative to the script's directory.
    script_dir = Path(__file__).resolve().parent
    catalog_path = Path(input_dir)  # this replaces the previously hard-coded catalog path
    output_dir = script_dir / testrun_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load hyperparameter configuration from train_configuration.txt.
    config = load_config()
    crop_shape = config.get("crop_shape", (128, 128))
    classification_shape = config.get("classification_shape", (512, 512))
    validation_split = config.get("validation_split", 0.2)
    batch_size = config.get("batch_size", 1)
    lr = config.get("lr", 1e-5)
    crop_epochs = config.get("crop_epochs", 25)
    classification_epochs = config.get("classification_epochs", 100)
    seed = config.get("seed", 42)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Define model names based on the testrun name.
    crop_model_name = f"{testrun_name}_crop_model"
    classification_model_name = f"{testrun_name}_classification_model"

    # Directories for saving training artifacts and datasets.
    crop_model_dir = output_dir / crop_model_name
    crop_train_dataset_path = output_dir / f"{crop_model_name}/image_dataset_{crop_model_name}.tfrecord"
    crop_val_dataset_path = output_dir / f"{crop_model_name}/mask_dataset_{crop_model_name}.tfrecord"

    classification_model_dir = output_dir / classification_model_name
    classification_train_dataset_path = output_dir / f"{classification_model_name}/train_dataset_{classification_model_name}.tfrecord"
    classification_val_dataset_path = output_dir / f"{classification_model_name}/val_dataset_{classification_model_name}.tfrecord"
    classification_enum_path = output_dir / f"{classification_model_name}/enums.json"

    # Initialize the models.
    crop_model = CropModel(crop_shape)
    classification_model = ClassificationModel(classification_shape)

    # === Crop Model Preparation ===
    try:
        crop_train_dataset = tf.data.Dataset.load(str(crop_train_dataset_path))
        crop_val_dataset = tf.data.Dataset.load(str(crop_val_dataset_path))
        print("=== Crop Model Datasets loaded ===")
    except Exception as e:
        print("=== Creating image pairs for Crop Model ===")
        pairs = construct_pairs(
            x_dir_path=catalog_path / "images",
            y_dir_path=catalog_path / "masks"
        )
        crop_train_pairs, crop_val_pairs = train_test_split(pairs, test_size=validation_split, random_state=seed)

        print("=== Creating datasets for Crop Model ===")
        crop_train_dataset = create_dataset(crop_train_pairs, batch_size, crop_shape)
        crop_val_dataset = create_dataset(crop_val_pairs, batch_size, crop_shape)

        print("=== Caching and Shuffling datasets ===")
        AUTOTUNE = tf.data.AUTOTUNE
        crop_train_dataset = crop_train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        crop_val_dataset = crop_val_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        print("=== Saving Crop Model datasets ===")
        crop_train_dataset.save(str(crop_train_dataset_path))
        crop_val_dataset.save(str(crop_val_dataset_path))

    # === Crop Model Training ===
    if not crop_model.load_model(crop_model_dir):
        print("=== Crop Model does not exist. Creating a new model... ===")
        crop_model.model = build_resnet34_model(input_shape=(*crop_shape, 3))
        crop_model.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(lr)
        )

    crop_model.train_model(
        train_dataset=crop_train_dataset,
        val_dataset=crop_val_dataset,
        num_epochs=crop_epochs,
        checkpoint_path=crop_model_dir
    )
    crop_model.save(crop_model_dir)

    crop_model.predict_dir(
        input_dir=catalog_path / "images",
        output_dir=catalog_path / "predict_masks",
        output_shape=classification_shape
    )
    input(f"Please clean up {catalog_path / 'predict_masks'} and press Enter to continue...")

    # === Classification Model Preparation ===
    try:
        train_dataset = tf.data.Dataset.load(str(classification_train_dataset_path))
        val_dataset = tf.data.Dataset.load(str(classification_val_dataset_path))
        with open(classification_enum_path, "r") as f:
            enumerations = json.load(f)
        print("=== Classification Model Datasets loaded ===")
    except Exception as e:
        print("=== Creating Classification Model datasets ===")
        enumerations = [str(coin.parts[-1]) for coin in get_directories(catalog_path / "images")]
        crop_predict_dir = str(catalog_path / "predict_masks")
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            crop_predict_dir,
            seed=seed,
            validation_split=0.2,
            subset='training',
            batch_size=batch_size,
            image_size=classification_shape
        )
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            crop_predict_dir,
            seed=seed,
            validation_split=0.2,
            subset='validation',
            batch_size=batch_size,
            image_size=classification_shape
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        print("=== Saving Classification Model datasets ===")
        train_dataset.save(str(classification_train_dataset_path))
        val_dataset.save(str(classification_val_dataset_path))
        with open(classification_enum_path, "w") as f:
            json.dump(enumerations, f, indent=4)

    # === Classification Model Training ===
    model_path = classification_model_dir / f"keras_{classification_model_name}.h5"

    if not classification_model.load_model(classification_model_dir):
        classification_model.model = build_conv_model(classification_shape, len(enumerations))

    classification_model.model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    classification_model.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=classification_epochs,
        checkpoint_dir=classification_model_dir
    )

    # Optionally, you can save the final model
    # classification_model.model.save(model_path)
