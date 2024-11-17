import faulthandler
import os
import sys

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QDialog

from core.utilities.helper import parse_directory_into_dictionary, qimage_to_cv2, cv2_to_qimage, transparent_to_hue, \
    apply_transformations, imgaug_transformation

catalog_path = "coin_catalog"


class ImagePopup(QDialog):
    def __init__(self, qimage, title="Image Preview", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        # Set up the layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create a QLabel to hold the image
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Convert QImage to QPixmap and set it to the QLabel
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

        # Optionally resize window to the image size
        self.resize(pixmap.width(), pixmap.height())

def show_image_popup(qimage):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Create and show the popup dialog
    popup = ImagePopup(qimage)
    popup.exec()


faulthandler.enable()
src_image = QImage("core/gui/pictures/default_image.jpg")
show_image_popup(src_image)
cv2_uncropped_image: np.ndarray = qimage_to_cv2( QImage("core/gui/pictures/default_image.jpg")) # memory violation
# cv2_uncropped_image: np.ndarray = qimage_to_cv2(src_image) # pass
image = cv2_to_qimage(cv2_uncropped_image)
show_image_popup(image)
