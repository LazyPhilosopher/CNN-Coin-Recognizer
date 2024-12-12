import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

from core.utilities.helper import get_directories


def load_png_as_tensor(file_path, shape):
    tensor = tf.io.read_file(file_path)
    tensor = tf.image.decode_image(tensor, channels=3)
    tensor.set_shape([None, None, 3])
    tensor = tf.image.resize(tensor, shape)
    tensor = tensor / 255
    return tensor


if __name__=="__main__":
    root_path = Path("D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/coin_catalog/augmented_30")
    catalog_path = Path(root_path, "predict_masks")

    """ Hyperparemeters """
    testrun_name = "crops_predict"
    image_shape = (512, 512)
    num_epochs = 15
    validation_split = 0.2
    batch_size = 32

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    """ Directory for storing files """
    if not os.path.exists(output_dir := Path(os.path.dirname(__file__), "trained")):
        os.makedirs(output_dir)

    enumerations = [str(coin.parts[-1]) for coin in get_directories(Path(catalog_path))]

    train_dataset_path = Path(output_dir, f"{testrun_name}/train_dataset_{testrun_name}.tfrecord")
    val_dataset_path = Path(output_dir, f"{testrun_name}/val_dataset_{testrun_name}.tfrecord")
    enum_path = f"trained/enumerations/train_enum_{testrun_name}.json"

    try:
        train_dataset = tf.data.Dataset.load(str(train_dataset_path))
        val_dataset = tf.data.Dataset.load(str(val_dataset_path))
        json.dump(enumerations, enum_path)
        print("=== Datasets loaded ===")

    except:
        """ Split dataset into train and validation subsets """
        train_dataset = tf.keras.utils.image_dataset_from_directory(catalog_path,
                                                               seed=42,
                                                               validation_split=0.2,
                                                               subset='training',
                                                               batch_size=batch_size,
                                                               image_size=image_shape)
        val_dataset = tf.keras.utils.image_dataset_from_directory(catalog_path,
                                                             seed=42,
                                                             validation_split=0.2,
                                                             subset='validation',
                                                             batch_size=batch_size,
                                                             image_size=image_shape)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        print("=== Save datasets ===")
        train_dataset.save(str(train_dataset_path))
        val_dataset.save(str(val_dataset_path))

    for images, labels in train_dataset.take(1):
        print(images.shape, labels.numpy())

    """ Model """
    model_path = Path(output_dir, f"{testrun_name}/keras_{testrun_name}.keras")
    if os.path.exists(model_path):
        print("Model already exists. Loading the model...")
        model = tf.keras.models.load_model(model_path)

    else:
        model = Sequential([
            # data_augmentation,
            layers.Rescaling(1. / 255),
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(enumerations))
        ])

    """ Training """
    model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=15, validation_data=val_dataset)

    model.save(model_path)