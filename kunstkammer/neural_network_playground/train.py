import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from core.utilities.helper import get_directories
from kunstkammer.neural_network_playground.classification import ClassificationModel
from kunstkammer.neural_network_playground.core.helper import construct_pairs, create_dataset
from kunstkammer.neural_network_playground.core.models import build_resnet34_model, build_conv_model
from kunstkammer.neural_network_playground.crop import CropModel

trained_model_dir = Path(os.path.dirname(__file__), "trained")
catalog_path = Path("D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/coin_catalog/augmented_50")
crop_shape = (128, 128)
classification_shape = (512, 512)
validation_split = 0.2
batch_size = 1
lr = 1e-5

crop_epochs = 25
classification_epochs = 100

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

crop_model_name = "remastered_crop_model_50"
classification_model_name = "remastered_classification_model_50"


if __name__ == "__main__":
    crop_model = CropModel(crop_shape)
    classification_model = ClassificationModel(classification_shape)

    """ Directory for storing files """
    if not os.path.exists(output_dir := Path(os.path.dirname(__file__), "trained")):
        os.makedirs(output_dir)

    crop_model_dir = Path(output_dir, f"{crop_model_name}")
    crop_train_dataset_path = Path(output_dir, f"{crop_model_name}/image_dataset_{crop_model_name}.tfrecord")
    crop_val_dataset_path = Path(output_dir, f"{crop_model_name}/mask_dataset_{crop_model_name}.tfrecord")

    classification_model_dir = Path(output_dir, f"{classification_model_name}")
    classification_train_dataset_path = Path(output_dir, f"{classification_model_name}/train_dataset_{classification_model_name}.tfrecord")
    classification_val_dataset_path = Path(output_dir, f"{classification_model_name}/val_dataset_{classification_model_name}.tfrecord")
    classification_enum_path = Path(output_dir, f"{classification_model_name}/enums.json")

    """ Crop Model Preparation """
    try:
        crop_train_dataset = tf.data.Dataset.load(str(crop_train_dataset_path))
        crop_val_dataset = tf.data.Dataset.load(str(crop_val_dataset_path))

        print("=== Crop Model Datasets loaded ===")

    except:
        print("=== Create image pairs ===")
        pairs = construct_pairs(x_dir_path=Path(catalog_path, "images"), y_dir_path=Path(catalog_path, "masks"))
        crop_train_pairs, crop_val_pairs = train_test_split(pairs, test_size=validation_split)

        print("=== Create datasets ===")
        crop_train_dataset = create_dataset(crop_train_pairs, batch_size, crop_shape)
        crop_val_dataset = create_dataset(crop_val_pairs, batch_size, crop_shape)

        print("=== Shuffle datasets ===")
        AUTOTUNE = tf.data.AUTOTUNE
        crop_train_dataset = crop_train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        crop_val_dataset = crop_val_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        print("=== Save datasets ===")
        crop_train_dataset.save(str(crop_train_dataset_path))
        # crop_train_dataset.save("D:/Projects/bachelor_thesis\OpenCV2-Coin-Recognizer/kunstkammer/neural_network_playground/trained/remastered_crop_model_30/test.tfrecord")
        crop_val_dataset.save(str(crop_val_dataset_path))

    """ Crop Model Training """
    if not crop_model.load_model(crop_model_dir):
        print("=== Crop Model does not exist. Creating a new model... ===")
        crop_model.model = build_resnet34_model(input_shape=(*crop_shape, 3))

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-3,
        #     decay_steps=1000,
        #     decay_rate=0.96
        # )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # crop_model.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        crop_model.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(lr)
        )

    crop_model.train_model(train_dataset=crop_train_dataset, val_dataset=crop_val_dataset, num_epochs=crop_epochs, checkpoint_path=crop_model_dir)
    crop_model.save(crop_model_dir)

    crop_model.predict_dir(input_dir=Path(catalog_path, "images"), output_dir=Path(catalog_path, "predict_masks"), output_shape=classification_shape)
    input(f"Please clean up {Path(catalog_path, 'predict_masks')}")

    """ Classification Model Preparation """
    try:
        # raise Exception
        train_dataset = tf.data.Dataset.load(str(classification_train_dataset_path))
        val_dataset = tf.data.Dataset.load(str(classification_val_dataset_path))
        with open(classification_enum_path, "r") as f:
            enumerations = json.load(f)
        print("=== Identification Model Datasets loaded ===")

    except:
        """ Split dataset into train and validation subsets """
        enumerations = [str(coin.parts[-1]) for coin in get_directories(Path(catalog_path, "images"))]
        crop_predict_dir = str(Path(catalog_path, "predict_masks"))
        train_dataset = tf.keras.utils.image_dataset_from_directory(crop_predict_dir,
                                                                    seed=42,
                                                                    validation_split=0.2,
                                                                    subset='training',
                                                                    batch_size=batch_size,
                                                                    image_size=classification_shape)
        val_dataset = tf.keras.utils.image_dataset_from_directory(crop_predict_dir,
                                                                  seed=42,
                                                                  validation_split=0.2,
                                                                  subset='validation',
                                                                  batch_size=batch_size,
                                                                  image_size=classification_shape)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        print("=== Save Classification Datasets ===")
        train_dataset.save(str(classification_train_dataset_path))
        val_dataset.save(str(classification_val_dataset_path))
        with open(classification_enum_path, "w") as f:
            json.dump(enumerations, f, indent=4)

    """ Classification Model Training """
    model_path = Path(output_dir, f"{classification_model_name}/keras_{classification_model_name}.h5")

    if not classification_model.load_model(classification_model_dir):
        classification_model.model = build_conv_model(classification_shape, len(enumerations))

    classification_model.model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    classification_model.train_model(train_dataset=train_dataset,
                                     val_dataset=val_dataset,
                                     epochs=classification_epochs,
                                     checkpoint_dir=classification_model_dir)

    # model.save(model_path)


    # knihovny
    # augmentace
    # pouzite site
    # dulezite casti kodu
    # zminka problemu
    #