import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.utilities.helper import get_directories


def threshold_to_black_and_white(image_tensor, threshold=0.004):
    """
    Converts an RGB tensor to a black-and-white tensor based on a threshold.
    Pixels where all channels are >= threshold are set to White (1.0, 1.0, 1.0).
    Otherwise, they are set to Black (0.0, 0.0, 0.0).

    Args:
        image_tensor: TensorFlow tensor of shape (height, width, 3) with RGB values.
        threshold: Float value for the threshold.

    Returns:
        A tensor of the same shape as the input with values set to either (1.0, 1.0, 1.0) or (0.0, 0.0, 0.0).
    """
    # Create a boolean mask where all channels are >= threshold
    mask = tf.reduce_all(image_tensor >= threshold, axis=-1)  # Shape: (height, width)

    # Create a white pixel tensor (1, 1, 1)
    # white_pixel = tf.constant([1, 1, 1], dtype=image_tensor.dtype)
    white_pixel = tf.constant([1, 1, 1], dtype=tf.uint8)

    # Create a black pixel tensor (0, 0, 0)
    # black_pixel = tf.constant([0, 0, 0], dtype=image_tensor.dtype)
    black_pixel = tf.constant([0, 0, 0], dtype=tf.uint8)

    # Apply the mask to choose between white and black
    result = tf.where(mask[..., tf.newaxis], white_pixel, black_pixel)

    return result


def apply_rgb_mask(image_tensor, mask_tensor):
    """
    Masks an RGB image with a binary RGB mask. Keeps original pixel values where the mask is white (1, 1, 1),
    and sets to black (0, 0, 0) where the mask is black (0, 0, 0).

    Args:
        image_tensor: TensorFlow tensor of shape (height, width, 3) representing the original RGB image.
        mask_tensor: TensorFlow tensor of shape (height, width, 3) representing the binary RGB mask
                     with values either (1, 1, 1) or (0, 0, 0).

    Returns:
        A TensorFlow tensor of the same shape as the input, masked by the binary mask.
    """
    # Ensure mask is binary (1s or 0s)
    mask_bool = tf.reduce_all(mask_tensor == 1, axis=-1, keepdims=True)  # Shape: (height, width, 1)

    # Use tf.where to apply the mask
    result = tf.where(mask_bool, image_tensor, tf.zeros_like(image_tensor))

    return result


def save_tensor_as_png(tensor, file_path, bit_depth=8):
    """
    Saves a floating-point tensor as a PNG file.

    Args:
        tensor: TensorFlow tensor of shape (height, width, channels) with floating-point values.
        file_path: String path to save the PNG file.
        bit_depth: Integer, bit depth for the PNG file (8 or 16).
    """
    # Ensure the tensor is in the range [0.0, 1.0]
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)

    # Scale tensor to the appropriate integer range
    if bit_depth == 8:
        tensor = tf.cast(tensor * 255.0, tf.uint8)  # Scale to [0, 255]
    elif bit_depth == 16:
        tensor = tf.cast(tensor * 65535.0, tf.uint16)  # Scale to [0, 65535]
    else:
        raise ValueError("bit_depth must be either 8 or 16")

    # Encode as PNG
    png_bytes = tf.io.encode_png(tensor)

    # Save to file
    tf.io.write_file(file_path, png_bytes)


def load_png_as_tensor(file_path, shape):
    tensor = tf.io.read_file(file_path)
    tensor = tf.image.decode_image(tensor, channels=3)
    tensor.set_shape([None, None, 3])
    tensor = tf.image.resize(tensor, shape)
    tensor = tensor / 255
    return tensor


def normalize_tensor(tensor):
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    normalized_tensor = (predicted_mask - min_val) / (max_val - min_val + 1e-8)
    # normalized_crop = image[0] * normalized_tensor * 255.0
    # normalized_tensor = tf.clip_by_value(normalized_crop, 0, 255)
    normalized_tensor = tf.cast(normalized_tensor, dtype=tf.uint8)
    return normalized_tensor


if __name__ == "__main__":
    catalog_path = Path("coin_catalog/augmented_30")
    mask_shape = (128, 128)
    output_shape = (512, 512)

    testrun_name = "crops_predict_ideal"

    # enum_path = f"trained/enumerations/train_enum_{testrun_name}.json"
    enumerations = [str(coin.parts[-1]) for coin in get_directories(Path(catalog_path, "images"))]

    background_deletion_model_path = Path("kunstkammer/background_deletion/trained/coin_full/keras_coin_full.keras")
    background_deletion_model = tf.keras.models.load_model(background_deletion_model_path)
    classification_model_path = Path("kunstkammer/object_recognition/trained/crops_predict/keras_crops_predict.keras")
    classification_model = tf.keras.models.load_model(classification_model_path)

    samples = [
        "coin_catalog/augmented_30/predict_masks/(Czech Republic, 50 Korun, 2008)/0_10.png",
        "coin_catalog/augmented_30/predict_masks/(Czech Republic, 1 Koruna, 2018)/1_28.png",
        "coin_catalog/augmented_30/predict_masks/(USA, 2.5 Dollar, 1909)/1_7.png",
        "coin_catalog/augmented_30/predict_masks/(Great britain, 0.5 Souvereign, 1906)/0_13.png",
        "coin_catalog/augmented_30/predict_masks/(Iran, 0.5 Souvereign, 1925)/4_3.png",
        "coin_catalog/augmented_30/predict_masks/(Austria-Hungary, 20 Korona, 1893)/0_8.png",
        "coin_catalog/augmented_30/predict_masks/(Czech Republic, 10 Korun, 2020)/0_3.png",
        "coin_catalog/augmented_30/predict_masks/(France, 2 Franc, 1917)/0_20.png",
        "coin_catalog/augmented_30/predict_masks/(Czech Republic, 5 Korun, 2002)/0_17.png",
        "coin_catalog/augmented_30/predict_masks/(India, 1 Rupee, 1840)/4_26.png",
        "coin_catalog/augmented_30/predict_masks/(Austria-Hungary, 20 Korona, 1893)/0_11.png",
    ]

    results = []
    for sample_path in samples:
        image_small = load_png_as_tensor(str(sample_path), mask_shape)
        image_small = tf.expand_dims(image_small, axis=0)  # Batch enpacking

        predicted_mask = background_deletion_model.predict(image_small)
        predicted_mask = predicted_mask[0][..., :3]
        bw_mask = threshold_to_black_and_white(predicted_mask, threshold=0.0035)
        bw_mask = tf.image.resize(bw_mask, output_shape)

        image_full = load_png_as_tensor(str(sample_path), output_shape)

        image_output = apply_rgb_mask(image_full, bw_mask)
        input_image_exp_dim = tf.expand_dims(image_output, 0)

        predictions = classification_model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions)

        sample_dir = Path(sample_path).parts[-2]
        predict_dir = enumerations[np.argmax(result)]

        results.append(f"[{'PASS' if sample_dir == predict_dir else 'FAIL'}] {sample_dir} -> {predict_dir}")

    [print(result) for result in results]
