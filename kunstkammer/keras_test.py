import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from random import random

import cv2
import numpy as np
import tensorflow as tf
from keras.src.applications.resnet_v2 import ResNet152V2
from keras.src.layers import GlobalAveragePooling2D, BatchNormalization
from keras.src.optimizers import Adam

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

from core.gui.ImageCollector import catalog_dir
from core.utilities.helper import parse_directory_into_dictionary, get_directories


def format_dataset(dataset_path):
    test_val_ratio = 0.8

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
        if random() < test_val_ratio:
            train_dict[(country, coin_name, year, prefix)] = catalog_dict[(country, coin_name, year, prefix)]
        else:
            val_dict[(country, coin_name, year, prefix)] = catalog_dict[(country, coin_name, year, prefix)]


    return enum_dict, train_dict, val_dict


def create_model(input_shape_full=(128, 128, 3), input_shape_hue=(128, 128, 1), num_classes=10):
    # Full image branch
    full_input = Input(shape=input_shape_full, name="full_image")
    x1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(full_input)
    x1 = MaxPooling2D(pool_size=2)(x1)
    x1 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=2)(x1)
    x1 = Flatten()(x1)

    # Hue image branch
    hue_input = Input(shape=input_shape_hue, name="hue_image")
    x2 = Conv2D(16, kernel_size=3, activation='relu', padding='same')(hue_input)
    x2 = MaxPooling2D(pool_size=2)(x2)
    x2 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = MaxPooling2D(pool_size=2)(x2)
    x2 = Flatten()(x2)

    # Concatenate branches
    combined = Concatenate()([x1, x2])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    # Create model
    model = Model(inputs=[full_input, hue_input], outputs=output)
    return model


class ResNet152V2Classifier:
    def __init__(self, full_input_shape=(224, 224, 3), hue_input_shape=(224, 224, 1), num_classes=14, learning_rate=0.001, train_base=False):
        """
        Initialize the ResNet152V2Classifier.

        Parameters:
            full_input_shape (tuple): Shape of input images (height, width, channels) for full image.
            hue_input_shape (tuple): Shape of input images (height, width, channels) for hue image.
            num_classes (int): Number of output classes.
            learning_rate (float): Learning rate for the optimizer.
            train_base (bool): If True, the base model is trainable for fine-tuning.
        """
        self.full_input_shape = full_input_shape
        self.hue_input_shape = hue_input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_base = train_base
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the ResNet152V2 model with a custom classification head for both full and HUE images.

        Returns:
            model (tf.keras.Model): The compiled model.
        """
        # Load the ResNet152V2 model pre-trained on ImageNet
        base_model = ResNet152V2(
            include_top=False,
            weights="imagenet",
            input_shape=self.full_input_shape
        )

        # Freeze the base model layers if train_base is False
        base_model.trainable = self.train_base

        # Define inputs for full and HUE images
        full_image = Input(shape=self.full_input_shape, name="full_image")
        hue_image = Input(shape=self.hue_input_shape, name="hue_image")

        # Process full image through the ResNet backbone
        full_x = base_model(full_image, training=not self.train_base)
        full_x = GlobalAveragePooling2D()(full_x)

        # Process hue image through simple convolution layers
        hue_x = Conv2D(32, (3, 3), padding="same", activation="relu")(hue_image)
        hue_x = BatchNormalization()(hue_x)
        hue_x = Conv2D(64, (3, 3), padding="same", activation="relu")(hue_x)
        hue_x = BatchNormalization()(hue_x)
        hue_x = GlobalAveragePooling2D()(hue_x)

        # Concatenate the features from both inputs
        combined = Concatenate()([full_x, hue_x])

        # Add custom dense layers
        x = Dense(512, activation="relu")(combined)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        # Create the full model with two inputs (full_image, hue_image) and one output
        model = Model(inputs={"full_image": full_image, "hue_image": hue_image}, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def get_model(self):
        """
        Returns the built Keras model.

        Returns:
            model (tf.keras.Model): The compiled model.
        """
        return self.model


def resize_image(image, target_size=(128, 128)):
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BILINEAR)


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


def create_dataset(data_dict, enum_dict, target_size=(128, 128)):
    full_images, hue_images, labels = [], [], []

    for idx, (key, entry) in enumerate(data_dict.items()):
        # Load and preprocess images
        full_image, hue_image = load_image_pair({'full': str(entry['full']), 'hue': str(entry['hue'])})

        # Resize images
        full_image = resize_image(full_image, target_size)
        hue_image = resize_image(hue_image, target_size)

        # Convert to uint8
        full_image = tf.cast(full_image * 255, tf.uint8)  # Assumes full_image is normalized (0, 1)
        hue_image = tf.cast(hue_image * 255, tf.uint8)  # Assumes hue_image is normalized (0, 1)

        # Append data
        full_images.append(full_image)
        hue_images.append(hue_image)
        labels.append(enum_dict[key[:3]])  # (Country, Coin, Year) key
        print(f"=== \r{idx+1}/{len(data_dict)} ===", end="")

    # Return dataset
    return tf.data.Dataset.from_tensor_slices(
        (
            {
                "full_image": tf.stack(full_images),  # Assign key 'full_image'
                "hue_image": tf.stack(hue_images)    # Assign key 'hue_image'
            },
            tf.constant(labels)  # The labels
        )
    )

def save_dataset(dataset, file_path):
    tf.data.experimental.save(
        dataset, file_path, compression='GZIP'
    )
    with open(file_path + '/element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
        pickle.dump(dataset.element_spec, out_)


def load_dataset(file_path):
    with open(file_path + '/element_spec', 'rb') as in_:
        es = pickle.load(in_)

    return tf.data.Dataset.load(
        file_path, es, compression='GZIP'
    )


if __name__ == "__main__":
    # catalog_path = Path("coin_catalog/augmented")
    catalog_path = Path("coin_catalog/augmented_micro")
    testrun_name = "ResNet512V2"
    shape = (224, 224)

    enumerations, training_dict, validation_dict = format_dataset(catalog_path)

    try:
        train_dataset = load_dataset(f"trained/datasets/train_dataset_{testrun_name}.tfrecord")
        val_dataset = load_dataset(f"trained/datasets/val_dataset_{testrun_name}.tfrecord")

    except:
        print("Training Dataset Creation:")
        train_dataset = create_dataset(training_dict, enumerations, target_size=shape)
        print("Validation Dataset Creation:")
        val_dataset = create_dataset(validation_dict, enumerations, target_size=shape)

        save_dataset(train_dataset, f"trained/datasets/train_dataset_{testrun_name}.tfrecord")
        save_dataset(train_dataset, f"trained/datasets/val_dataset_{testrun_name}.tfrecord")

    print("Train Dataset Shuffle")
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
    print("Validation Dataset Shuffle")
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


    print("Creating Model")
    model_path = f"trained/models/keras_{testrun_name}.keras"
    if not os.path.exists(model_path):
        print("Model does not exist. Creating a new model...")
        # model = create_model(num_classes=len(enumerations))
        resnet512 = ResNet152V2Classifier(num_classes=len(enumerations))

        resnet512.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model = resnet512.get_model()

        print("Training Model")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20
        )
        model.save(model_path)

    else:
        print("Model already exists. Loading the model...")
        model = tf.keras.models.load_model(model_path)


    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

