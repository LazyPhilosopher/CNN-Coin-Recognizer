import json
import os
from collections import defaultdict
from pathlib import Path
from random import random

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

from core.gui.ImageCollector import catalog_dir
from core.utilities.helper import parse_directory_into_dictionary, get_directories



def format_dataset():
    catalog_path = Path("coin_catalog/augmented")

    test_val_ratio = 0.8

    enum_dict = {}
    catalog_dict = defaultdict(dict)
    for country_dir in get_directories(catalog_path):
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


# def load_and_preprocess_image(image_path, is_gray=False):
#     # Load and normalize images
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (128, 128))  # Resize all images to the same size
#     image = image / 255.0  # Normalize to [0, 1]
#     # if not is_gray:
#     #     image = np.expand_dims(image, axis=0)  # Add channel dimension
#     return image


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


def resize_image(image, target_size=(128, 128)):
    """
    Resize an image to the target size (default: 128x128).

    Args:
        image (tf.Tensor): The input image tensor.
        target_size (tuple): The desired size (height, width).

    Returns:
        tf.Tensor: The resized image tensor.
    """
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BILINEAR)


def load_image_pair(entry):
    """
    Load and preprocess the image pair (full-color and hue) from the given dictionary entry.
    """
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
    """
    Create a TensorFlow dataset with resized images, returning two inputs: full and hue.

    Args:
        data_dict (dict): Dictionary of data paths.
        enum_dict (dict): Enumeration dictionary for labels.
        target_size (tuple): Target image size (height, width).

    Returns:
        tf.data.Dataset: The dataset with ((full, hue), label) structure.
    """
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
    return tf.data.Dataset.from_tensor_slices(((tf.stack(full_images), tf.stack(hue_images)), tf.constant(labels)))


def save_dataset(dataset, file_path):
    """
    Save the dataset as a TFRecord file with two inputs: full and hue.
    """

    dir_path, filename = os.path.split(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with tf.io.TFRecordWriter(file_path) as writer:
        for (full_image, hue_image), label in dataset:
            example = tf.train.Example(features=tf.train.Features(feature={
                'full_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(full_image).numpy()])),
                'hue_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(hue_image).numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())


def load_saved_dataset(file_path):
    """
    Load a dataset from a TFRecord file with two inputs: full and hue.
    """
    if not tf.io.gfile.exists(file_path):
        raise FileNotFoundError(f"TFRecord file not found: {file_path}")

    raw_dataset = tf.data.TFRecordDataset(file_path)

    def parse_example(example_proto):
        feature_description = {
            'full_image': tf.io.FixedLenFeature([], tf.string),
            'hue_image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        full_image = tf.image.decode_png(parsed_features['full_image'], channels=3)
        hue_image = tf.image.decode_png(parsed_features['hue_image'], channels=1)
        label = parsed_features['label']

        return (full_image, hue_image), label

    return raw_dataset.map(parse_example)


if __name__ == "__main__":
    enumerations, training_dict, validation_dict = format_dataset()

    try:
        train_dataset = load_saved_dataset('trained/datasets/train_dataset.tfrecord')
        val_dataset = load_saved_dataset('trained/datasets/val_dataset.tfrecord')

    except:
        print("Training Dataset Creation:")
        train_dataset = create_dataset(training_dict, enumerations)
        print("Validation Dataset Creation:")
        val_dataset = create_dataset(validation_dict, enumerations)

        save_dataset(train_dataset, "trained/datasets/train_dataset.tfrecord")
        save_dataset(train_dataset, "trained/datasets/val_dataset.tfrecord")

    print("Train Dataset Shuffle")
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
    print("Validation Dataset Shuffle")
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    print("Creating Model")
    model = create_model(num_classes=len(enumerations))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training Model")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20
    )

    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    model.save('trained/models/keras_test.h5')
