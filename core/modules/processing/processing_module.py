import os

import imgaug.augmenters as iaa

from PySide6.QtCore import QObject, QThread
from PySide6.QtWidgets import QApplication
from numpy import ndarray

from core.qt_communication.base import *
from core.qt_communication.messages.processing_module.Requests import *
from core.qt_communication.messages.processing_module.Responses import *
from core.utilities.helper import qimage_to_cv2, remove_background_rembg, cv2_to_qimage, \
    parse_directory_into_dictionary, transparent_to_hue


class ProcessingModule(QObject):

    def __init__(self):
        super().__init__()

        self.is_running = False
        self.is_processing = False
        self.main_thread = QThread()
        self.qt_signals = CommonSignals()
        self.qt_signals.processing_module_request.connect(self.handle_request)

    def start_process(self):
        self.moveToThread(self.main_thread)
        self.main_thread.started.connect(self.worker)
        self.main_thread.start()
        self.is_running = True

    def worker(self):
        while self.is_running:
            QApplication.processEvents()

    def handle_request(self, request: MessageBase):
        request_handlers = {
            RemoveBackgroundRequest: self._handle_remove_background,
            AugmentCoinCatalogRequest: self._handle_catalog_augmentation_request
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            if self.is_processing:
                return

            self.is_processing = True
            handler(request)
            self.is_processing = False

    def _handle_remove_background(self, request: RemoveBackgroundRequest):
        image = qimage_to_cv2(request.picture)
        img_no_bg = remove_background_rembg(image)
        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=cv2_to_qimage(img_no_bg),
                                   source=Modules.PROCESSING_MODULE,
                                   destination=request.source))

    def _handle_catalog_augmentation_request(self, request: AugmentCoinCatalogRequest):
        catalog_dict = parse_directory_into_dictionary(request.catalog_path)
        os.makedirs(os.path.join(request.catalog_path, "augmented"), exist_ok=True)
        for country in catalog_dict.keys():
            for coin_name in catalog_dict[country].keys():
                for year in catalog_dict[country][coin_name].keys():
                    os.makedirs(os.path.join(request.catalog_path, country, coin_name, year), exist_ok=True)

                    for coin_photo in catalog_dict[country][coin_name][year]["uncropped"]:

                        if not coin_photo in catalog_dict[country][coin_name][year]["cropped"]:
                            continue

                        cropped_coin_photo_path = os.path.join(request.catalog_path, country, coin_name, year,
                                                               "cropped", coin_photo)
                        uncropped_coin_photo_path = os.path.join(request.catalog_path, country, coin_name, year,
                                                               "uncropped", coin_photo)
                        pass

                        cv2_uncropped_image: ndarray = qimage_to_cv2(QImage(uncropped_coin_photo_path))
                        cv2_cropped_image: ndarray = qimage_to_cv2(QImage(cropped_coin_photo_path))
                        cv2_cropped_hue: QImage = transparent_to_hue(cv2_cropped_image)
                        pass
                        #
                        # seq = iaa.Sequential([
                        #     # Apply local distortions with random scale between 0.005 and 0.02
                        #     iaa.PiecewiseAffine(scale=(0.005, 0.02)),
                        #     iaa.Affine(
                        #         scale={
                        #             "x": (0.8, 1.2),  # Random scaling along the x-axis
                        #             "y": (0.8, 1.2)  # Random scaling along the y-axis
                        #         },
                        #         rotate=(-90, 90)  # Random rotation between -90 and 90 degrees
                        #     ),
                        #     iaa.Affine(scale=(0.25, 1.2)),  # Scale between 0.5x and 1.2x
                        #     # Apply Gaussian blur with a random sigma between 0.4 and 0.5
                        #     iaa.GaussianBlur(sigma=(0, 10))
                        # ])
                        #
                        # noise = iaa.Sequential([
                        #     # Add Gaussian noise with random scale between 0 and 0.05*255
                        #     iaa.AdditiveGaussianNoise(scale=(0.1 * 255, 0.5 * 255))
                        # ])
                        #
                        # for i in range(10):
                        #     deterministic_seq = seq.to_deterministic()
                        #     deterministic_noise = noise.to_deterministic()
                        #
                        #     cv2_full_picture = deterministic_seq.augment_image(cv2_uncropped_image)
                        #     cv2_full_picture = deterministic_noise.augment_image(cv2_full_picture)
                        #     cv2_full_image: QImage = cv2_to_qimage(cv2_full_picture)
                        #     # full_image = full_image.convertToFormat(QImage.Format_RGB888)
                        #     cv2_full_pixmap: QPixmap = QPixmap.fromImage(cv2_full_image)
                        #     cv2_full_pixmap.save(
                        #         os.path.join(request.catalog_path, "augmented", country, coin_name, year,
                        #                      f"{i}_full.png"))
                        #
                        #     cv2_hue_picture = deterministic_seq.augment_image(cv2_cropped_hue)
                        #     cv2_hue_picture = deterministic_noise.augment_image(cv2_hue_picture)
                        #     cv2_hue_image: QImage = cv2_to_qimage(cv2_hue_picture)
                        #     # full_image = full_image.convertToFormat(QImage.Format_RGB888)
                        #     cv2_hue_pixmap: QPixmap = QPixmap.fromImage(cv2_hue_image)
                        #     cv2_hue_pixmap.save(
                        #         os.path.join(request.catalog_path, "augmented", country, coin_name, year,
                        #                      f"{i}_hue.png"))

