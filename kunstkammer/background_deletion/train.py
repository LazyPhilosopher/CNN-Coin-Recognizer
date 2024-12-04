import json
import os
from pathlib import Path

from core.utilities.helper import get_directories

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import build_model

global image_shape
global enumerations


def load_image(image_path, size, add_aplha=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # For RGB images
    image = tf.image.resize(image, size)

    if add_aplha:
        alpha_channel = tf.ones_like(image[..., :1])
        image_with_alpha = tf.concat([image, alpha_channel], axis=-1)
        image_with_alpha = tf.image.resize(image_with_alpha, size)
        image_with_alpha = image_with_alpha / 255.0
        return image_with_alpha

    image = image / 255.0
    return image


def process_pair(input_path, output_path):
    input_image = load_image(input_path, image_shape)
    output_image = load_image(output_path, image_shape, add_aplha=True)
    return input_image, output_image


def create_dataset(pairs, batch_size):
    input_paths, output_paths = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(input_paths), list(output_paths)))
    dataset = dataset.map(lambda x, y: process_pair(x, y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def construct_pairs(x_dir_path: Path, y_dir_path: Path):
    pairs = []
    for class_name in enumerations:
        input_class_dir = os.path.join(x_dir_path, class_name)
        output_class_dir = os.path.join(y_dir_path, class_name)
        if os.path.isdir(input_class_dir) and os.path.isdir(output_class_dir):
            input_images = sorted(os.listdir(input_class_dir))
            output_images = sorted(os.listdir(output_class_dir))
            for img_name in input_images:
                if img_name in output_images:  # Match input-output pairs
                    pairs.append((
                        os.path.join(input_class_dir, img_name),
                        os.path.join(output_class_dir, img_name)
                    ))
    return pairs


if __name__ == "__main__":
    catalog_path = Path("coin_catalog/augmented")

    """ Hyperparemeters """
    image_shape = (128, 128)

    testrun_name = "test"
    num_epochs = 20
    validation_split = 0.2
    batch_size = 1
    lr = 3e-4

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    """ Directory for storing files """
    if not os.path.exists(output_dir := Path(os.path.dirname(__file__), "trained")):
        os.makedirs(output_dir)

    model_path = Path(output_dir, f"keras_{testrun_name}.keras")
    train_dataset_path = Path(output_dir, f"image_dataset_{testrun_name}.tfrecord")
    val_dataset_path = Path(output_dir, f"mask_dataset_{testrun_name}.tfrecord")

    try:
        train_dataset = tf.data.Dataset.load(str(train_dataset_path))
        val_dataset = tf.data.Dataset.load(str(val_dataset_path))

        print("=== Datasets loaded ===")

    except:
        print("=== Create image pairs ===")
        enumerations = [str(coin.stem) for coin in get_directories(Path(catalog_path, "images"))]
        pairs = construct_pairs(x_dir_path = Path(catalog_path, "images"), y_dir_path=Path(catalog_path, "masks"))
        train_pairs, val_pairs = train_test_split(pairs, test_size=validation_split)

        print("=== Create datasets ===")
        train_dataset = create_dataset(train_pairs, batch_size)
        val_dataset = create_dataset(val_pairs, batch_size)

        print("=== Shuffle datasets ===")
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        print("=== Save datasets ===")
        train_dataset.save(str(train_dataset_path))
        val_dataset.save(str(val_dataset_path))

    """ Model """
    if not os.path.exists(model_path):
        print("Model does not exist. Creating a new model...")
        model = build_model(input_shape=(*image_shape, 3))
        model.compile(
            loss="binary_crossentropy",  # Or categorical_crossentropy depending on your problem
            optimizer=tf.keras.optimizers.Adam(lr),
            metrics=["accuracy"]  # Optional: Add accuracy or other metrics to monitor during training
        )
    else:
        print("Model already exists. Loading the model...")
        model = tf.keras.models.load_model(model_path)

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=num_epochs,
              callbacks=callbacks
              )

    model.save(model_path)