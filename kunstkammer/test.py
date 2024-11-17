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


def _handle_catalog_augmentation_request():
    catalog_dict = parse_directory_into_dictionary(catalog_path)
    os.makedirs(os.path.join(catalog_path, "augmented"), exist_ok=True)

    for country in catalog_dict.keys():
        for coin_name in catalog_dict[country].keys():
            for year in catalog_dict[country][coin_name].keys():
                os.makedirs(os.path.join(catalog_path, country, coin_name, year), exist_ok=True)

                for coin_photo in catalog_dict[country][coin_name][year]["uncropped"]:

                    if not coin_photo in catalog_dict[country][coin_name][year]["cropped"]:
                        continue

                    cropped_coin_photo_path = os.path.join(catalog_path, country, coin_name, year,
                                                           "cropped", coin_photo)
                    uncropped_coin_photo_path = os.path.join(catalog_path, country, coin_name, year,
                                                             "uncropped", coin_photo)

                    os.makedirs(os.path.join(catalog_path, "augmented", country, coin_name, year),
                                exist_ok=True)

                    uncropped_image = QImage(uncropped_coin_photo_path)
                    cropped_image = QImage(cropped_coin_photo_path)

                    cv2_uncropped_image = qimage_to_cv2(uncropped_image)
                    cv2_cropped_image = qimage_to_cv2(cropped_image)

                    # cv2_hue_image = transparent_to_hue(cv2_cropped_image)

                    # Transformations
                    for i in range(1):
                        cv2_uncropped_image, cv2_croped_image = (
                            imgaug_transformation(full_image=cv2_uncropped_image, hue_image=cv2_cropped_image))

                        full_image = cv2_to_qimage(cv2_uncropped_image)
                        cv2_hue_image = transparent_to_hue(cv2_croped_image)
                        croped_image = cv2_to_qimage(cv2_hue_image)

                        full_image.save(
                            os.path.join(catalog_path, "augmented", country, coin_name, year,
                                         f"{os.path.splitext(coin_photo)[0]}_{i}_full.png"))
                        croped_image.save(
                            os.path.join(catalog_path, "augmented", country, coin_name, year,
                                         f"{os.path.splitext(coin_photo)[0]}_{i}_hue.png"))

faulthandler.enable()
_handle_catalog_augmentation_request()