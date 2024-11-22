import json
import os
from collections import defaultdict
from random import random

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

from core.gui.ImageCollector import catalog_dir
from core.utilities.helper import parse_directory_into_dictionary, get_directories



def format_dataset():
    catalog_path = "./coin_catalog/augmented"

    test_val_ratio = 0.8

    enum_dict = {}
    catalog_dict = defaultdict(dict)
    for country in get_directories(catalog_path):
        for coin_name in get_directories(os.path.join(catalog_path, country)):
            for year in get_directories(coin_dir := os.path.join(catalog_path, country, coin_name)):
                enum_dict[(country, coin_name, year)] = len(enum_dict)

                for filename in os.listdir(year_dir := os.path.join(coin_dir, year)):
                    if filename.endswith("_full.png") or filename.endswith("_hue.png"):
                        prefix = "_".join(filename.split("_")[:2])

                        if filename.endswith("_full.png"):
                            catalog_dict[(country, coin_name, year, prefix)]["full"] = os.path.join(year_dir, filename)
                        elif filename.endswith("_hue.png"):
                            catalog_dict[(country, coin_name, year, prefix)]["hue"] = os.path.join(year_dir, filename)

    train_dict = defaultdict(dict)
    val_dict = defaultdict(dict)

    for country, coin_name, year, prefix in catalog_dict.keys():
        if random() < test_val_ratio:
            train_dict[(country, coin_name, year, prefix)] = catalog_dict[(country, coin_name, year, prefix)]
        else:
            val_dict[(country, coin_name, year, prefix)] = catalog_dict[(country, coin_name, year, prefix)]


    return enum_dict, train_dict, val_dict


def load_and_preprocess_image(image_path, is_gray=False):
    # Load and normalize images
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))  # Resize all images to the same size
    image = image / 255.0  # Normalize to [0, 1]
    # if not is_gray:
    #     image = np.expand_dims(image, axis=0)  # Add channel dimension
    return image


def create_dataset(data_dict, enumerations):
    images_full, images_hue, labels = [], [], []

    for idx, (key, paths) in enumerate(data_dict.items()):
        # Load images
        full_img = load_and_preprocess_image(paths["full"])
        hue_img = load_and_preprocess_image(paths["hue"], is_gray=True)

        images_full.append(full_img)
        images_hue.append(hue_img)

        # Map class to label
        labels.append(enumerations[key[:3]])
        print(f"\r{idx}/{len(data_dict)}", end="")


    # Convert to TensorFlow tensors
    images_full = tf.convert_to_tensor(images_full, dtype=tf.float32)
    images_hue = tf.convert_to_tensor(images_hue, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    return tf.data.Dataset.from_tensor_slices(({"full_image": images_full, "hue_image": images_hue, "labels": labels}))


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


def create_example(full_image, hue_image, labels):
    # Ensure the images are of type uint8 (scaling if necessary)
    full_image = tf.cast(tf.clip_by_value(full_image * 255, 0, 255), tf.uint8)
    hue_image = tf.cast(tf.clip_by_value(hue_image * 255, 0, 255), tf.uint8)
    hue_image = tf.expand_dims(hue_image, axis=-1)

    # Encode the images as PNG
    full_bytes = tf.io.encode_png(full_image)
    hue_bytes = tf.io.encode_png(hue_image)

    # Creating a feature dictionary to store as Example
    feature = {
        'full_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[full_bytes.numpy()])),
        'hue_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hue_bytes.numpy()])),
        'label': tf.convert_to_tensor(labels, dtype=tf.int32)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def parse_example(serialized_example):
    feature_description = {
        'full_image': tf.io.FixedLenFeature([], tf.string),
        'hue_image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    # Decode the images
    full_image = tf.image.decode_png(parsed['full_image'], channels=3)
    hue_image = tf.image.decode_png(parsed['hue_image'], channels=1)
    label = parsed['label']

    return {'full_image': full_image, 'hue_image': hue_image}, label


if __name__ == "__main__":
    enumerations, training_dict, validation_dict = format_dataset()

    print("Training Dataset Creation:")
    # try:
    #     raw_dataset = tf.data.TFRecordDataset('trained/datasets/dataset.tfrecord')
    #     train_dataset = raw_dataset.map(parse_example)
    #
    #     for features, label in train_dataset.take(1):
    #         print("Full image shape:", features['full_image'].shape)
    #         print("Hue image shape:", features['hue_image'].shape)
    #         print("Label:", label.numpy())
    # except:
    train_dataset = create_dataset(training_dict, enumerations)
    with tf.io.TFRecordWriter('trained/datasets/dataset.tfrecord') as writer:
        for data in train_dataset:
            example = create_example(data["full_image"], data["hue_image"], data["labels"])  # Convert image to serialized example
            writer.write(example.SerializeToString())

    print("Validation Dataset Creation:")
    try:
        raw_dataset = tf.data.TFRecordDataset('trained/datasets/validation_dataset.tfrecord')
        train_dataset = raw_dataset.map(parse_example)

        for features, label in train_dataset.take(1):
            print("Full image shape:", features['full_image'].shape)
            print("Hue image shape:", features['hue_image'].shape)
            print("Label:", label.numpy())
    except:
        val_dataset = create_dataset(validation_dict, enumerations)
        with tf.io.TFRecordWriter('trained/datasets/validation_dataset.tfrecord') as writer:
            for image, label in train_dataset:
                example = create_example(image, label)
                writer.write(example.SerializeToString())

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
