import json
import os
from pathlib import Path

from core.helper import construct_pairs, threshold_to_black_and_white, save_tensor_as_png

from core.utilities.helper import get_directories, get_files, apply_rgb_mask

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


class CropModel:

    def __init__(self, input_shape=tuple):
        self.input_shape = input_shape
        self.model = None

    def load_model(self, model_testrun_dir: Path | str):
        try:
            testrun_name = model_testrun_dir.parts[-1]
            full_model_path = str(Path(model_testrun_dir, f"keras_{testrun_name}.h5"))
            self.model = tf.keras.models.load_model(full_model_path)
            print(f"=== Successfully loaded model: {full_model_path} ===")
            return True
        except:
            return False

    def save(self, model_testrun_dir):
        testrun_name = model_testrun_dir.parts[-1]
        full_model_path = str(Path(model_testrun_dir, f"keras_{testrun_name}.h5"))
        self.model.save(full_model_path)

    def create_train_val_datasets(self, dataset_path: Path | str, validation_split: float = 0.2):
        dataset_path = str(dataset_path)
        pairs = construct_pairs(x_dir_path=Path(dataset_path, "images"), y_dir_path=Path(dataset_path, "masks"))
        train_pairs, val_pairs = train_test_split(pairs, test_size=validation_split)

    def load_image(self, image_path: Path | str, resize_shape=None):
        if not resize_shape:
            resize_shape = self.input_shape

        image_path = str(image_path)

        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, resize_shape)
        image = image / 255
        return image

    def predict_mask(self, batched_image, threshold: float = 0.003):
        predicted_mask = self.model.predict(batched_image)
        predicted_mask = predicted_mask[0][..., :3]
        return threshold_to_black_and_white(predicted_mask, threshold=threshold)

    def train_model(self, checkpoint_path, train_dataset, val_dataset, num_epochs):
        full_checkpoint_path = Path(checkpoint_path, f"keras_{checkpoint_path.parts[-1]}.h5")

        callbacks = [
            ModelCheckpoint(full_checkpoint_path, monitor='loss', verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
            EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)
        ]

        return self.model.fit(train_dataset,
                  validation_data=val_dataset,
                  epochs=num_epochs,
                  callbacks=callbacks
                  )

    def predict_dir(self, input_dir, output_dir, output_shape):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_path in get_files(directory_path=input_dir):
            image = self.load_image(file_path)
            batched_image = tf.expand_dims(image, 0)
            bw_mask = self.predict_mask(batched_image)
            bw_mask = tf.image.resize(bw_mask, output_shape)

            image_full = self.load_image(str(file_path), output_shape)

            masked_image = apply_rgb_mask(image_full, bw_mask)

            output_path = Path(output_dir, file_path.parts[-1])
            save_tensor_as_png(masked_image, str(output_path))

        for directory in get_directories(input_dir):
            self.predict_dir(input_dir=Path(input_dir, directory.parts[-1]),
                             output_dir=Path(output_dir, directory.parts[-1]),
                             output_shape=output_shape)