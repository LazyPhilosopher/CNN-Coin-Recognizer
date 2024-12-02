import json
import os
from collections import defaultdict
from pathlib import Path
from random import random

import tensorflow as tf
from tensorflow.keras import layers, models

from core.utilities.helper import get_directories, resize_image


def build_detection_model(resolution=128):
    input_image = tf.keras.layers.Input(shape=(resolution, resolution, 3), name="full_image")

    # Backbone for feature extraction
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(input_image)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Output the bounding box (normalized coordinates: [y_min, x_min, y_max, x_max])
    bounding_box = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(x)

    return tf.keras.Model(inputs=input_image, outputs=bounding_box, name="detection_model")


# def create_model(resolution=(128, 128), num_classes=10):
#     """
#     Creates a small CNN model for simultaneous detection and classification.
#
#     Args:
#     - resolution: Tuple, resolution of the input images (e.g., (128, 128)).
#     - num_classes: Integer, the number of unique coin classes.
#
#     Returns:
#     - TensorFlow Keras Model.
#     """
#     input_image = layers.Input(shape=(resolution[0], resolution[1], 3), name="full_image")
#     input_hue = layers.Input(shape=(resolution[0], resolution[1], 1), name="hue_image")
#
#     # Shared Backbone
#     x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_image)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
#     backbone_features = layers.MaxPooling2D((2, 2), name="backbone_features")(x)
#
#     # Detection Branch
#     detection = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation="relu", padding="same")(
#         backbone_features)
#     detection = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same")(detection)
#     detection = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation="relu", padding="same")(detection)
#     detection_output = layers.Conv2D(1, (1, 1), activation="sigmoid", name="detection_output")(detection)
#
#     # Classification Branch
#     classification = layers.GlobalAveragePooling2D()(backbone_features)
#     classification = layers.Dense(128, activation="relu")(classification)
#     classification = layers.Dropout(0.3)(classification)
#     classification_output = layers.Dense(num_classes, activation="softmax", name="classification_output")(
#         classification)
#
#     # Model
#     model = models.Model(inputs=[input_image, input_hue], outputs=[detection_output, classification_output])
#     return model


def load_image_pair(entry):
    full_image = tf.io.read_file(entry['full'])
    full_image = tf.image.decode_png(full_image, channels=3)  # Decode as RGB
    full_image = tf.image.convert_image_dtype(full_image, tf.float32)  # Normalize to [0, 1]
    full_image = resize_image(full_image)

    hue_image = tf.io.read_file(entry['hue'])
    hue_image = tf.image.decode_png(hue_image, channels=1)  # Decode as grayscale
    hue_image = tf.image.convert_image_dtype(hue_image, tf.float32)  # Normalize to [0, 1]
    hue_image = resize_image(hue_image)

    return full_image, hue_image


def format_dataset(dataset_path: Path, train_val_ratio=0.8):
    enum_dict = {}
    catalog_dict = defaultdict(dict)
    for country_dir in get_directories(dataset_path):
        for coin_dir in get_directories(country_dir):
            for year_dir in get_directories(coin_dir):
                enum_dict[(country := country_dir.parts[-1],
                           coin_name := coin_dir.parts[-1],
                           year := year_dir.parts[-1])] = len(enum_dict)

                for filepath in year_dir.iterdir():
                    filename = filepath.parts[-1]
                    if filename.endswith(suffix := "full.png") or filename.endswith(suffix := "hue.png"):
                        # prefix = "_".join(filename.split("_")[:2])
                        prefix = filename[:-(len(suffix)+1)]

                        if suffix == "full.png":
                            catalog_dict[(country, coin_name, year, prefix)]["full"] = filepath
                        elif suffix == "hue.png":
                            catalog_dict[(country, coin_name, year, prefix)]["hue"] = filepath

    train_dict = defaultdict(dict)
    val_dict = defaultdict(dict)

    for country, coin_name, year, prefix in catalog_dict.keys():
        if random() < train_val_ratio:
            train_dict[(country, coin_name, year, prefix)] = catalog_dict[(country, coin_name, year, prefix)]
        else:
            val_dict[(country, coin_name, year, prefix)] = catalog_dict[(country, coin_name, year, prefix)]


    return enum_dict, train_dict, val_dict


def create_dataset(data_dict, enum_dict, target_size=(128, 128)):
    full_images, hue_images, labels = [], [], []

    for idx, (key, entry) in enumerate(data_dict.items()):
        full_image, hue_image = load_image_pair({'full': str(entry['full']), 'hue': str(entry['hue'])})

        full_image = resize_image(full_image, target_size)
        hue_image = resize_image(hue_image, target_size)

        full_image = tf.cast(full_image * 255, tf.uint8)
        hue_image = tf.cast(hue_image * 255, tf.uint8)

        full_images.append(full_image)
        hue_images.append(hue_image)
        labels.append(enum_dict[key[:3]])
        print(f"\r=== {idx+1}/{len(data_dict)} ===", end="")
    print("\n")

    # Return dataset
    return tf.data.Dataset.from_tensor_slices(
        (
            {
                "full_image": tf.stack(full_images),
                "hue_image": tf.stack(hue_images)
            },
            tf.constant(labels)  # The labels
        )
    )


def save_dataset(dataset, file_path):
    dataset.save(file_path)


if __name__ == "__main__":
    catalog_path = Path("coin_catalog/augmented")
    testrun_name = "round2"
    shape = (128, 128)

    model_path = f"trained/models/keras_{testrun_name}.keras"
    train_dataset_path = f"trained/datasets/train_dataset_{testrun_name}.tfrecord"
    val_dataset_path = f"trained/datasets/val_dataset_{testrun_name}.tfrecord"
    enum_path = f"trained/enumerations/train_enum_{testrun_name}.json"

    try:
        train_dataset = tf.data.Dataset.load(train_dataset_path)
        val_dataset = tf.data.Dataset.load(val_dataset_path)
        with open(enum_path) as f:
            enumerations = json.load(f)

    except:
        enumerations, training_dict, validation_dict = format_dataset(dataset_path=catalog_path, train_val_ratio=0.8)

        print("Training Dataset Creation:")
        train_dataset = create_dataset(training_dict, enumerations, target_size=shape)
        print("Validation Dataset Creation:")
        val_dataset = create_dataset(validation_dict, enumerations, target_size=shape)
        print("Enumeration JSON Creation:")
        enumerations_inv = {value: key for key, value in enumerations.items()}

        train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        save_dataset(train_dataset, train_dataset_path)
        save_dataset(val_dataset, val_dataset_path)
        with open(enum_path, "w") as f:
            json.dump(enumerations_inv, f, indent=4)

    if not os.path.exists(model_path):
        print("Model does not exist. Creating a new model...")

        detection_model = build_detection_model(resolution=128)
        detection_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        detection_model.fit(train_dataset, validation_data=val_dataset, epochs=25)

        val_loss, val_accuracy = detection_model.evaluate(val_dataset)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy}")

        # model.save(model_path)

    else:
        print("Model already exists. Loading the model...")
        model = tf.keras.models.load_model(model_path)

    # val_loss, val_accuracy = model.evaluate(val_dataset)
    # print(f"Validation Loss: {val_loss}")
    # print(f"Validation Accuracy: {val_accuracy}")
