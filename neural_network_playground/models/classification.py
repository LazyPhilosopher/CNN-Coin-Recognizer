import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import compute_class_weight

# from core.helper import construct_pairs
#
# from core.utilities.helper import get_directories, get_files


class ClassificationModel:
    def __init__(self, input_shape=tuple):
        self.input_shape = input_shape
        self.model = None

    def load_model(self, model_testrun_dir: Path | str):
        try:
            testrun_name = model_testrun_dir.parts[-1]
            full_model_path = str(Path(model_testrun_dir, f"keras_{testrun_name}.keras"))
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
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=2e-4,
            decay_steps=100000,
            decay_rate=0.92,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

        self.model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])


        full_checkpoint_path = Path(checkpoint_dir, f"keras_{checkpoint_dir.parts[-1]}.keras")

        callbacks = [
            ModelCheckpoint(full_checkpoint_path, monitor='val_accuracy', verbose=1,
                            save_best_only=True),
            # ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
        ]

        y_train = tf.concat([label_batch for _, label_batch in train_dataset.unbatch()], axis=0).numpy()

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )

        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

        return self.model.fit(train_dataset,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset,
                            class_weight=class_weights_dict)
