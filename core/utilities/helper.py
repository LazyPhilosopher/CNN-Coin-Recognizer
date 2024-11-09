import shutil
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMessageBox, QApplication, QWidget, QDialog, QLabel, QPushButton, QVBoxLayout
from rembg import remove

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
        for country in out_dict.keys():
            country_path = os.path.join(dir_path, country)
            out_dict[country] = {coin_name: {} for coin_name in get_directories(country_path)}
            for coin in out_dict[country].keys():
                coin_path = os.path.join(dir_path, country, coin)
                out_dict[country][coin] = {year: [] for year in get_directories(coin_path)}
        return out_dict
    except:
        return None


def remove_background_rembg(img):
    img = remove(img)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def cv2_to_qimage(cv_img):
    # Check if the array is in RGB or RGBA format
    if cv_img.ndim == 3 and cv_img.shape[2] == 3:  # RGB
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    elif cv_img.ndim == 3 and cv_img.shape[2] == 4:  # RGBA
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    else:
        raise ValueError("Unsupported array shape for RGB(A) format")

    # The QImage does not take ownership of the array data by default,
    # so we need to make a deep copy if the data might change
    return qimage.copy()


def qimage_to_cv2(qimage):
    width = qimage.width()
    height = qimage.height()

    # Check the format of the QImage
    format = qimage.format()

    # Handle images with an alpha channel (transparency)
    if format in (QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied, QImage.Format_RGBA8888):
        channels = 4
        ptr = qimage.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, channels))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)  # Convert RGBA to OpenCV's BGRA format

    # Handle images without an alpha channel
    elif format == QImage.Format_RGB32:
        channels = 4  # QImage stores RGB32 as ARGB (premultiplied alpha)
        ptr = qimage.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, channels))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)  # Drop alpha channel, convert to BGR

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
