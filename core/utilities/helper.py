import shutil
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMessageBox, QApplication, QWidget, QDialog, QLabel, QPushButton, QVBoxLayout

import win32com.client


def move_files(file_list, source_folder, destination_folder, create_dir=True):
    # Ensure destination folder exists if create_dir is True
    if create_dir and not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in file_list:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        if os.path.exists(source_path):
            # Move the file to the new destination
            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} -> {destination_path}")
        else:
            print(f"File not found: {source_path}")


def show_confirmation_dialog(parent: QWidget, title: str, message: str):
    # Create a message box
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Question)  # Use the question icon
    msg_box.setWindowTitle(title)
    msg_box.setText(message)

    # Add Yes and No buttons using QMessageBox.StandardButton
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    msg_box.setDefaultButton(QMessageBox.StandardButton.No)  # Set No as the default button

    # Display the dialog and capture the user's response
    result = msg_box.exec()

    if result == QMessageBox.StandardButton.Yes:
        return True  # User clicked Yes
    else:
        return False  # User clicked No


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


def convert_pixmap_to_grayscale(pixmap: QPixmap):
    # Step 1: Convert QPixmap to QImage
    image = pixmap.toImage()

    # Step 2: Convert QImage to grayscale (QImage.Format_Grayscale8)
    grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)

    # Step 3: Convert the grayscale QImage back to QPixmap
    grayscale_pixmap = QPixmap.fromImage(grayscale_image)

    return grayscale_pixmap


def qpixmap_to_numpy(pixmap: QPixmap):
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()

    ## Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))

    return img


def numpy_to_qimage(data) -> QImage:
    height, width = data.shape
    return QImage(data, width, height, QImage.Format_Grayscale8)


def remove_background(img, contours):
    # Create a mask for the background (single channel)
    mask = np.zeros_like(img[:, :, 0])

    if contours:
        cnt = max(contours, key=cv2.contourArea)  # Find the largest contour
        cv2.drawContours(mask, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)  # Fill the contour

    # Convert the mask to a 4-channel version (for RGBA)
    mask_alpha = cv2.merge([mask, mask, mask, mask])

    # Create a transparent image
    img_with_alpha = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    # Set background pixels to transparent where the mask is zero
    img_with_alpha[mask_alpha[:, :, 3] == 0] = [0, 0, 0, 0]  # Set to transparent (RGBA)

    return img_with_alpha


def cv2_to_qimage(cv_img):
    # Ensure that the image is contiguous in memory
    cv_img = np.ascontiguousarray(cv_img)

    # Extract height, width, and number of channels from the image
    height, width, channels = cv_img.shape
    bytes_per_line = channels * width

    # Choose the appropriate QImage format based on the number of channels
    if channels == 4:  # BGRA to RGBA
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

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


def show_image_popup(image):
    """
    Display the given QImage in a popup window.

    Args:
    image (QImage): The QImage to display in the popup.
    """
    # Create a dialog window
    dialog = QDialog()
    dialog.setWindowTitle("Image Preview")
    dialog.setFixedSize(image.width(), image.height())

    # Convert QImage to QPixmap for display
    pixmap = QPixmap.fromImage(image)

    # Create a label to display the image
    label = QLabel(dialog)
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)

    # Add a button to close the dialog
    close_button = QPushButton("Close", dialog)
    close_button.clicked.connect(dialog.accept)

    # Set layout for the dialog
    layout = QVBoxLayout(dialog)
    layout.addWidget(label)
    layout.addWidget(close_button)

    # Show the dialog
    dialog.exec()

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
