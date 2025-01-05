import json
import os
from pathlib import Path
import tensorflow as tf

def get_directories(directory_path: Path):
    return [entry for entry in directory_path.iterdir() if entry.is_dir()]

def construct_pairs(x_dir_path: Path, y_dir_path: Path):
    pairs = []

    enumerations = [str(coin.parts[-1]) for coin in get_directories(Path(x_dir_path))]
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

def load_enumerations(enum_path: Path | str):
    with open(os.path.join(enum_path, "enums.json"), "r") as f:
        enumerations = json.load(f)
    return enumerations


def load_image(image_path, size, add_aplha=False, is_mask=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # For RGB images
    image = tf.image.resize(image, size)

    if add_aplha:
        alpha_channel = tf.ones_like(image[..., :1])
        image_with_alpha = tf.concat([image, alpha_channel], axis=-1)
        image_with_alpha = tf.image.resize(image_with_alpha, size)
        image_with_alpha = image_with_alpha / 255.0
        return image_with_alpha

    # if is_mask:
    #     image = tf.image.convert_image_dtype(image, tf.float32)

    image = image / 255.0
    return image


def create_dataset(pairs, batch_size, image_shape):
    def process_pair(input_path, output_path):
        input_image = load_image(input_path, image_shape)
        # output_image = load_image(output_path, image_shape, add_aplha=True)
        output_image = load_image(output_path, image_shape, is_mask=True)
        return input_image, output_image

    input_paths, output_paths = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(input_paths), list(output_paths)))
    dataset = dataset.map(lambda x, y: process_pair(x, y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


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
