import os

from PySide6.QtCore import QObject, QThread
from PySide6.QtWidgets import QApplication

from core.qt_communication.base import *
from core.qt_communication.messages.processing_module.Requests import *
from core.qt_communication.messages.processing_module.Responses import *
from core.utilities.helper import qimage_to_cv2, remove_background_rembg, cv2_to_qimage, \
    parse_directory_into_dictionary, transparent_to_hue, imgaug_transformation


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
        cv2_img_no_bg = cv2_to_qimage(img_no_bg)
        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=cv2_img_no_bg,
                                   source=Modules.PROCESSING_MODULE,
                                   destination=request.source))

    def _handle_catalog_augmentation_request(self, request: AugmentCoinCatalogRequest):
        catalog_path = request.catalog_path
        catalog_dict = parse_directory_into_dictionary(catalog_path)
        os.makedirs(os.path.join(request.catalog_path, "augmented"), exist_ok=True)

        for country in catalog_dict.keys():
            for coin_name in catalog_dict[country].keys():
                for year in catalog_dict[country][coin_name].keys():
                    os.makedirs(os.path.join(request.catalog_path, country, coin_name, year), exist_ok=True)

                    for coin_photo in catalog_dict[country][coin_name][year]["uncropped"]:

                        if not coin_photo in catalog_dict[country][coin_name][year]["cropped"]:
                            continue

                        cropped_coin_photo_path = os.path.join(catalog_path, country, coin_name, year, "cropped")
                        uncropped_coin_photo_path = os.path.join(catalog_path, country, coin_name, year, "uncropped")

                        os.makedirs(os.path.join(catalog_path, "augmented", country, coin_name, year),
                                    exist_ok=True)

                        cropped_file_list = catalog_dict[country][coin_name][year]["cropped"]
                        uncpped_file_list = catalog_dict[country][coin_name][year]["uncropped"]
                        picture_filename_list = [picture for picture in uncpped_file_list if picture in cropped_file_list]

                        uncropped_qimage_files = [QImage(os.path.join(uncropped_coin_photo_path, picture)) for picture in picture_filename_list]
                        cropped_qimage_files = [QImage(os.path.join(cropped_coin_photo_path, picture)) for picture in picture_filename_list]

                        # uncropped_image = QImage(uncropped_coin_photo_path)
                        # cropped_image = QImage(cropped_coin_photo_path)

                        uncropped_cv2_files = [qimage_to_cv2(uncropped_image) for uncropped_image in uncropped_qimage_files]
                        cropped_cv2_files = [qimage_to_cv2(cropped_image) for cropped_image in cropped_qimage_files]
                        cropped_cv2_hue_files = [transparent_to_hue(cv2_cropped_image) for cv2_cropped_image in cropped_cv2_files]

                        # cv2_uncropped_image = qimage_to_cv2(uncropped_image)
                        # cv2_cropped_image = qimage_to_cv2(cropped_image)

                        # cv2_hue_image = transparent_to_hue(cv2_cropped_image)

                        # Transformations
                        for i in range(10):
                            cv2_warped_uncropped_image, cv2_warped_hue_image = (
                                imgaug_transformation(full_image_list=uncropped_cv2_files,
                                                      hue_image_list=cropped_cv2_hue_files))

                            augmented_qimage_list = [cv2_to_qimage(image) for image in cv2_warped_uncropped_image]
                            augmented_hue_list = [cv2_to_qimage(image) for image in cv2_warped_hue_image]

                            for image in augmented_qimage_list:
                                image.save(os.path.join(catalog_path, "augmented", country, coin_name, year,
                                                        f"{os.path.splitext(coin_photo)[0]}_{i}_full.png"))

                            for image in augmented_hue_list:
                                image.save(os.path.join(catalog_path, "augmented", country, coin_name, year,
                                             f"{os.path.splitext(coin_photo)[0]}_{i}_hue.png"))
