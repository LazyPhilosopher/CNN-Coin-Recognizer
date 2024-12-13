import json
import os
from pathlib import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from core.helper import construct_pairs

from core.utilities.helper import get_directories, get_files


class ClassificationModel:
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

    def load_image(self, image_path: Path | str):
        image_full = tf.keras.utils.load_img(image_path, target_size=self.input_shape)
        return tf.keras.utils.img_to_array(image_full)

    def predict(self, image, verbose=False):
        return self.model.predict(image, verbose=verbose)

    def train_model(self, checkpoint_dir, train_dataset, val_dataset, epochs):
        full_checkpoint_path = Path(checkpoint_dir, f"keras_{checkpoint_dir.parts[-1]}.h5")

        callbacks = [
            ModelCheckpoint(full_checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
        ]

        return self.model.fit(train_dataset,
                       epochs=epochs,
                       validation_data=val_dataset,
                       callbacks=callbacks)
