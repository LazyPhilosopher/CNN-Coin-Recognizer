import shutil
import os
import imgaug.augmenters as iaa
import imgaug as ia

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMessageBox, QApplication, QWidget, QDialog, QLabel, QPushButton, QVBoxLayout
from rembg import remove

from skimage import transform, filters, util
from skimage.util import random_noise

# import win32com.client


def get_directories(directory_path: str):
    return [entry for entry in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, entry))]


def get_files(directory_path: str):
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]


def create_coin_directory(catalog_path: str, coin_country: str, coin_name: str, coin_year: str):
    os.makedirs(os.path.join(catalog_path, coin_country, coin_name, coin_year, "cropped"), exist_ok=True)
    os.makedirs(os.path.join(catalog_path, coin_country, coin_name, coin_year, "uncropped"), exist_ok=True)


# def move_files(file_list, source_folder, destination_folder, create_dir=True):
#     # Ensure destination folder exists if create_dir is True
#     if create_dir and not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     for file_name in file_list:
#         source_path = os.path.join(source_folder, file_name)
#         destination_path = os.path.join(destination_folder, file_name)
#
#         if os.path.exists(source_path):
#             # Move the file to the new destination
#             shutil.move(source_path, destination_path)
#             print(f"Moved: {source_path} -> {destination_path}")
#         else:
#             print(f"File not found: {source_path}")


# def show_confirmation_dialog(parent: QWidget, title: str, message: str):
#     # Create a message box
#     msg_box = QMessageBox(parent)
#     msg_box.setIcon(QMessageBox.Icon.Question)  # Use the question icon
#     msg_box.setWindowTitle(title)
#     msg_box.setText(message)
#
#     # Add Yes and No buttons using QMessageBox.StandardButton
#     msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
#     msg_box.setDefaultButton(QMessageBox.StandardButton.No)  # Set No as the default button
#
#     # Display the dialog and capture the user's response
#     result = msg_box.exec()
#
#     if result == QMessageBox.StandardButton.Yes:
#         return True  # User clicked Yes
#     else:
#         return False  # User clicked No



# def remove_background(img, contours):
#     # Create a mask for the background (single channel)
#     mask = np.zeros_like(img[:, :, 0])
#
#     if contours:
#         cnt = max(contours, key=cv2.contourArea)  # Find the largest contour
#         cv2.drawContours(mask, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)  # Fill the contour
#
#     # Convert the mask to a 4-channel version (for RGBA)
#     mask_alpha = cv2.merge([mask, mask, mask, mask])
#
#     # Create a transparent image
#     img_with_alpha = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
#
#     # Set background pixels to transparent where the mask is zero
#     img_with_alpha[mask_alpha[:, :, 3] == 0] = [0, 0, 0, 0]  # Set to transparent (RGBA)
#
#     return img_with_alpha


def parse_directory_into_dictionary(dir_path: str):
    try:
        out_dict = {country: {} for country in get_directories(dir_path)}
        out_dict.pop("augmented", None)

        for country in out_dict.keys():
            country_path = os.path.join(dir_path, country)
            out_dict[country] = {coin_name: {} for coin_name in get_directories(country_path)}
            for coin in out_dict[country].keys():
                coin_path = os.path.join(dir_path, country, coin)
                out_dict[country][coin] = {year: {
                    "uncropped": get_files(os.path.join(coin_path, year, "uncropped")),
                    "cropped": get_files(os.path.join(coin_path, year, "cropped"))
                } for year in get_directories(coin_path)}
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

def imgaug_transformation(full_image: np.ndarray, hue_image: np.ndarray, seed: int = 42):
    # Ensure the images are contiguous arrays (important for memory layout)
    full_image = np.ascontiguousarray(full_image)
    hue_image = np.ascontiguousarray(hue_image)

    # Set a deterministic random state for reproducibility
    random_state = np.random.RandomState(seed)

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

    # Apply deterministic transformations with the same random state
    seq_common_det = seq_common.to_deterministic()

    # Apply the same deterministic transformation to both images
    full_image_aug = seq_common_det.augment_image(full_image)
    hue_image_aug = seq_common_det.augment_image(hue_image)

    return full_image_aug, hue_image_aug

# def show_image_popup(image):
#     """
#     Display the given QImage in a popup window.
#
#     Args:
#     image (QImage): The QImage to display in the popup.
#     """
#     # Create a dialog window
#     dialog = QDialog()
#     dialog.setWindowTitle("Image Preview")
#     dialog.setFixedSize(image.width(), image.height())
#
#     # Convert QImage to QPixmap for display
#     pixmap = QPixmap.fromImage(image)
#
#     # Create a label to display the image
#     label = QLabel(dialog)
#     label.setPixmap(pixmap)
#     label.setAlignment(Qt.AlignCenter)
#
#     # Add a button to close the dialog
#     close_button = QPushButton("Close", dialog)
#     close_button.clicked.connect(dialog.accept)
#
#     # Set layout for the dialog
#     layout = QVBoxLayout(dialog)
#     layout.addWidget(label)
#     layout.addWidget(close_button)
#
#     # Show the dialog
#     dialog.exec()

# def on_usb_device_connected():
#     print("A USB device has been connected.")
#
#
# def monitor_usb_devices():
#     wmi = win32com.client.GetObject("winmgmts:")
#     watcher = wmi.ExecNotificationQuery(
#         "SELECT * FROM Win32_DeviceChangeEvent"
#     )
#
#     while True:
#         event = watcher.NextEvent()
#         if event:
#             on_usb_device_connected()
