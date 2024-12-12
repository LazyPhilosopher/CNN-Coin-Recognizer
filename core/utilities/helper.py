import os
from pathlib import Path

import os
from pathlib import Path

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QTabWidget
from rembg import remove
from skimage import transform, filters, util
from skimage.util import random_noise


# import win32com.client


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



def get_directories(directory_path: Path):
    return [entry for entry in directory_path.iterdir() if entry.is_dir()]


def get_files(directory_path: Path):
    return [f for f in directory_path.iterdir() if f.is_file()]


def create_coin_directory(catalog_path: str, coin_country: str, coin_name: str, coin_year: str):
    os.makedirs(os.path.join(catalog_path, coin_country, coin_name, coin_year, "cropped"), exist_ok=True)
    os.makedirs(os.path.join(catalog_path, coin_country, coin_name, coin_year, "uncropped"), exist_ok=True)



def parse_directory_into_dictionary(dir_path: Path):
    try:
        out_dict = {country.parts[-1]: {} for country in get_directories(dir_path)}

        pop_keys = []
        for directory in out_dict:
            if "augmented" in directory:
                pop_keys.append(directory)
        [out_dict.pop(key) for key in pop_keys]

        for country in out_dict.keys():
            country_path = Path(dir_path / country)
            out_dict[country] = {coin_dir.parts[-1]: {} for coin_dir in get_directories(country_path)}
            for coin in out_dict[country].keys():
                coin_path = Path(dir_path / country / coin)
                out_dict[country][coin] = {year_dir.parts[-1]: {
                    "uncropped": get_files(Path(year_dir / "uncropped")),
                    "cropped": get_files(Path(year_dir / "cropped"))
                } for year_dir in get_directories(coin_path)}
        return out_dict
    except:
        return None


def remove_background_rembg(img):
    img = remove(img)
    # return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def cv2_to_qimage(cv_img):
    # Check if the array is in RGB or RGBA format
    if cv_img.ndim == 3 and cv_img.shape[2] == 3:  # RGB
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        cv_img_contig = np.ascontiguousarray(cv_img)
        qimage = QImage(cv_img_contig.data, width, height, bytes_per_line, QImage.Format_RGB888)
    elif cv_img.ndim == 3 and cv_img.shape[2] == 4:  # RGBA
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        cv_img_contig = np.ascontiguousarray(cv_img)
        qimage = QImage(cv_img_contig.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    else:
        raise ValueError("Unsupported array shape for RGB(A) format")
    return qimage


def qimage_to_cv2(qimage):
    width = qimage.width()
    height = qimage.height()

    # Check the format of the QImage
    format = qimage.format()

    # Handle images with an alpha channel (transparency)
    if format in (QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied, QImage.Format_RGBA8888):
        channels = 4
        ptr = qimage.bits()
        return np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, channels))
        # return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)  # Convert RGBA to OpenCV's BGRA format

    # Handle images without an alpha channel
    elif format == QImage.Format_RGB32:
        channels = 4  # QImage stores RGB32 as ARGB (premultiplied alpha)
        ptr = qimage.bits()
        return np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, channels))
        # return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Drop alpha channel, convert to BGR

    elif format in (QImage.Format_RGB888, QImage.Format_Indexed8):
        channels = 3
        ptr = qimage.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, channels))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert RGB to OpenCV's BGR format

    else:
        raise ValueError(f"Unsupported QImage format: {format}")


def crop_vertices_mask_from_image(image: QImage, vertices) -> QImage:
    ndarray = qimage_to_cv2(image)
    points = np.array([[point.x(), point.y()] for point in vertices], dtype=np.int32)

    mask = np.full(ndarray.shape[:2], 255, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 0)

    img_with_alpha = cv2.cvtColor(ndarray, cv2.COLOR_RGB2RGBA)
    img_with_alpha[mask == 0] = [0, 0, 0, 0]
    return cv2_to_qimage(img_with_alpha)


def transparent_to_hue(image):
    if image.shape[2] == 4:
        # Split the channels (B, G, R, A)
        b, g, r, a = cv2.split(image)

        # Create a mask where white is not transparent and black is where it's transparent
        mask = cv2.merge([a, a, a])  # Use alpha channel to create the mask (3-channel)

        # Create a blank hue image
        hue_image = cv2.merge([b, g, r])

        # Set black color where the mask is transparent (alpha = 0)
        mask_inverted = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(hue_image, hue_image, mask=mask_inverted[:, :, 0])

        # Set the remaining area to white (where it's not transparent)
        white_background = np.full_like(hue_image, 255)
        result_with_white_bg = cv2.add(result, white_background, mask=mask[:, :, 0])

        return result_with_white_bg
    else:
        raise ValueError("The image does not have an alpha channel!")


def apply_transformations(full_image, hue_image):
    # Define random parameters for transformations
    angle = np.random.uniform(-30, 30)  # Random rotation angle
    sigma = np.random.uniform(1, 3)  # Random blur sigma
    distort_strength = np.random.uniform(0.8, 1.2)  # Slight distortion
    noise_amount = np.random.uniform(0.02, 0.05)  # Random white noise level

    # 1. Apply rotation
    image1_rotated = transform.rotate(full_image, angle, resize=False, mode='edge')
    image2_rotated = transform.rotate(hue_image, angle, resize=False, mode='edge')

    # 2. Apply Gaussian blur
    image1_blurred = filters.gaussian(image1_rotated, sigma=sigma)
    image2_blurred = filters.gaussian(image2_rotated, sigma=sigma)

    # 3. Apply slight distortion (scale transformation)
    tform = transform.AffineTransform(scale=(distort_strength, distort_strength))
    # image1_distorted = transform.warp(image1_blurred, tform.inverse, mode='edge')
    # image2_distorted = transform.warp(image2_blurred, tform.inverse, mode='edge')

    # 4. Add random white noise
    image1_noisy = random_noise(image1_blurred, mode='gaussian', var=noise_amount**2)
    # image2_noisy = random_noise(image2_distorted, mode='gaussian', var=noise_amount**2)

    # Rescale from [0,1] to [0,255] if needed
    image1_final = util.img_as_ubyte(np.clip(image1_noisy, 0, 1))
    image2_final = util.img_as_ubyte(np.clip(image2_blurred, 0, 1))

    return image1_final, image2_final

def imgaug_transformation(image: np.ndarray, mask: np.ndarray, transparent: np.ndarray):
    # Ensure the images are contiguous arrays (important for memory layout)
    # contiguous_image_list = [np.ascontiguousarray(full_image.copy()) for full_image in full_image_list]
    # contiguous_hue_list = [np.ascontiguousarray(hue_image.copy()) for hue_image in hue_image_list]

    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    transparent = np.ascontiguousarray(transparent)

    # Set a deterministic random state for reproducibility

    # Initialize the augmentation sequence
    seq_common = iaa.Sequential(
        [
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # nearest neighbour or bilinear interpolation
                mode=ia.ALL  # use all scikit-image warping modes
            ),
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),  # Local deformation
        ],
        random_order=False  # Ensure the same order of augmentations
    )

    seq_noise = iaa.Sequential([
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Gaussian noise with random intensity
            iaa.SaltAndPepper(p=(0.01, 0.05)),  # Salt and pepper noise with random proportion
            iaa.AdditivePoissonNoise(lam=(0, 8)),  # Poisson noise with random lambda
        ])
    ])

    # Apply deterministic transformations with the same random state
    seq_common_det = seq_common.to_deterministic()
    seq_noise_det = seq_noise.to_deterministic()

    # Apply the same deterministic transformation to both images
    temp_image = seq_common_det.augment_image(image)
    image_aug = seq_noise_det.augment_image(temp_image)
    temp_crop = seq_common_det.augment_image(transparent)
    mask_aug = seq_common_det.augment_image(mask)
    crop_aug = seq_noise_det.augment_image(temp_crop[:, :, :3])
    crop_aug = apply_rgb_mask(crop_aug, mask_aug)
    crop_aug = np.concatenate([crop_aug, np.expand_dims(temp_crop[:, :, 3], axis=-1)], axis=-1)

    # crop_aug = tf.where(mask_aug[:, :, 3:4] == 0, tf.zeros_like(crop_aug), crop_aug)

    return image_aug, mask_aug, crop_aug


def get_tab_index_by_label(tab_widget: QTabWidget, label: str) -> int:
    for index in range(tab_widget.count()):
        if tab_widget.tabText(index) == label:
            return index
    return -1

def resize_image(image, target_size=(128, 128)):
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BILINEAR)


def transparent_to_mask(image):
    """
    Converts a transparent image (ndarray with an alpha channel) to a mask.
    Transparent pixels are set to (0, 0, 0), and non-transparent pixels are set to (1, 1, 1).

    Parameters:
        image (ndarray): Input image with shape (H, W, 4), where the last channel is the alpha channel.

    Returns:
        mask (ndarray): Output mask with shape (H, W, 3), where values are (0, 0, 0) or (1, 1, 1).
    """
    if image.shape[-1] != 4:
        raise ValueError("Input image must have an alpha channel (shape should be HxWx4).")

    # Extract the alpha channel
    alpha_channel = image[..., 3]

    # Create a binary mask: 1 for non-transparent pixels, 0 for transparent pixels
    binary_mask = np.where(alpha_channel > 0, 1, 0).astype(np.uint8)

    # Expand the binary mask to 3 channels (H, W, 3)
    mask = np.repeat(binary_mask[..., np.newaxis], 3, axis=-1)

    return mask


def load_png_as_tensor(file_path, shape):
    tensor = tf.io.read_file(file_path)
    tensor = tf.image.decode_image(tensor, channels=3)
    tensor.set_shape([None, None, 3])
    tensor = tf.image.resize(tensor, shape)
    tensor = tensor / 255
    return tensor


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