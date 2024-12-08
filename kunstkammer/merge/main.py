import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model

from core.utilities.helper import get_directories


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
    # output_image = load_image(output_path, image_shape, add_aplha=True)
    output_image = load_image(output_path, image_shape)
    return input_image, output_image

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


def create_dataset(pairs, batch_size):
    input_paths, output_paths = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(input_paths), list(output_paths)))
    dataset = dataset.map(lambda x, y: process_pair(x, y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def make_tensor_bw(tensor, treshold=0.25):
    # Convert the RGB values to grayscale (brightness) using the luminosity method
    brightness = np.dot(tensor[...,:3], [0.2989, 0.5870, 0.1140])
    mask = brightness >= treshold

    rgb_out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_out[mask] = [255, 255, 255]
    return rgb_out



image_shape = (128, 128)
batch_size = 1
lr = 1e-3
num_epochs = 20
validation_split = 0.2
catalog_path = Path("coin_catalog/augmented")

testrun_name = "merge"
if not os.path.exists(output_dir := Path(os.path.dirname(__file__), "trained")):
    os.makedirs(output_dir)

model_path = Path(output_dir, f"{testrun_name}/keras_{testrun_name}.keras")
crop_dataset_path = Path(output_dir, f"{testrun_name}/image_dataset_{testrun_name}.tfrecord")

# Load the pre-trained models
crop_model = tf.keras.models.load_model('D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/kunstkammer/background_deletion/trained/phase1/keras_phase1.keras')
classify_model = tf.keras.models.load_model('D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/kunstkammer/object_recognition/trained/object_recognition_ResNet50.keras')

# Freeze the layers if no further training is needed
crop_model.trainable = False
classify_model.trainable = False


enumerations = [str(coin.stem) for coin in get_directories(Path(catalog_path, "images"))]
pairs = construct_pairs(x_dir_path = Path(catalog_path, "images"), y_dir_path=Path(catalog_path, "images"))
crop_list = []
crop_labels = []

try:
    crop_dataset = tf.data.Dataset.load(str(Path(catalog_path, "images")))
    print("=== Dataset loaded ===")

except:
    image_dataset = tf.keras.utils.image_dataset_from_directory(str(Path(catalog_path, "images")),
                                                seed=42,
                                                batch_size=batch_size,
                                                image_size=image_shape)

    for idx, (image_batch, label_batch) in image_dataset.enumerate():
        crop_mask = crop_model.predict(image_batch)
        crop_mask = crop_mask[0]  # Remove batch dimension
        crop_mask = np.clip(crop_mask * 255, 0, 255).astype(np.uint8)
        crop_mask = make_tensor_bw(crop_mask, treshold=0.1)

        crop_labels.append(label_batch)
        crop_list.append(crop_mask*image_batch)
        print(f"{idx}/{len(image_dataset)}")

    crop_dataset = tf.data.Dataset.from_tensor_slices((crop_list, crop_labels))
    crop_dataset.save(str(crop_dataset_path))

dataset_size = tf.data.experimental.cardinality(crop_dataset).numpy()
val_size = int(dataset_size * validation_split)
train_size = dataset_size - val_size
train_dataset = crop_dataset.take(train_size)
val_dataset = crop_dataset.skip(train_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

""" Training """
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=1000,  # Adjust this based on dataset size and batch size
    decay_rate=0.96,  # Factor by which to decay
    staircase=True  # If True, the learning rate decays in discrete steps
)

classify_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(model_path, save_best_only=True),
    # ReduceLROnPlateau(factor=0.1, patience=5),
    EarlyStopping(patience=10)
]

history = classify_model.fit(
    train_ds,
    validation_data=val_dataset,
    epochs=num_epochs,
    callbacks=callbacks
)

classify_model.save(model_path)