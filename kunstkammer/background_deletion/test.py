import os
from pathlib import Path

import numpy as np
import tensorflow as tf


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


if __name__ == "__main__":
    catalog_path = Path("coin_catalog/augmented")
    mask_shape = (128, 128)
    output_shape = (512, 512)


    testrun_name = "coin_20"


    """ Directory for storing files """
    if not os.path.exists(output_dir := Path(os.path.dirname(__file__), "predict")):
        os.makedirs(output_dir)

    model_path = Path(os.path.dirname(__file__), f"trained/{testrun_name}/keras_{testrun_name}.h5")
    model = tf.keras.models.load_model(model_path)

    # input_path = "coin_catalog/France/2 Franc/1917/uncropped/1.png"
    # input_path = "coin_catalog/augmented_200/images/(Czech Republic, 1 Koruna, 2018)/2_100.png"
    # input_path = "coin_catalog/augmented_200/images/(USA, 2.5 Dollar, 1909)/0_108.png"
    input_path = "coin_catalog/augmented_30/images/(Czech Republic, 1 Koruna, 2018)/1_28.png"
    # input_path = "D:/Projects/bachelor_thesis/Background-Removal-using-Deep-Learning/people_segmentation/images/adult-attractive-full-body-41215.jpg"
    x = tf.io.read_file(input_path)
    x = tf.image.decode_image(x, channels=3)
    x.set_shape([None, None, 3])
    x = tf.image.resize(x, mask_shape)
    x = x / 255
    x_batch = tf.expand_dims(x, axis=0) # Batch enpacking

    y = model.predict(x_batch)
    y = y[0]  # Remove batch dimension
    y = y[:, :, :3]
    # y = np.clip(y * 255, 0, 255).astype(np.uint8)

    bw_mask = threshold_to_black_and_white(y, threshold=0.0035)
    bw_mask = tf.image.resize(bw_mask, output_shape)
    image_full = tf.io.read_file(input_path)
    image_full = tf.image.decode_image(image_full, channels=3)
    image_full.set_shape([None, None, 3])
    image_full = image_full / 255
    image_full = tf.image.resize(image_full, output_shape)

    # Save the image using TensorFlow's image encoding
    output = apply_rgb_mask(image_full, bw_mask)
    save_tensor_as_png(output, output_path := str(Path(output_dir, "predict", "output_image.png")))
    print(f"Image saved to {output_path}")
