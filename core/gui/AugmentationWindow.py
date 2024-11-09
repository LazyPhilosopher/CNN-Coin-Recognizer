import os

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMainWindow

from core.gui.pyqt6_designer.d_augmentation_window import Ui_AugmentationWindow
from core.qt_communication.base import *
from core.qt_communication.messages.processing_module.Requests import AugmentCoinCatalogRequest
from core.qt_communication.messages.processing_module.Responses import AugmentedPictureResponse


class AugmentationWindow(QMainWindow, Ui_AugmentationWindow):
    def __init__(self):
        super().__init__()
        # self.is_running = False
        self.setupUi(self)
        self._window: QMainWindow | None = None
        # self.main_thread = QThread()

        self.generate_augmented_data_button.clicked.connect(self.handle_request_augmented_data_button)

    def handle_request(self, request: MessageBase):
        request_handlers = {
            AugmentedPictureResponse: self._handle_augmented_picture_response
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)


    def handle_request_augmented_data_button(self):
        catalog_dir_path = "coin_catalog"
        rotation = self.rotation_slider.value()
        distortion = self.distortion_slider.value()
        blur = self.blur_slider.value()
        noise = self.noise_slider.value()
        picture_amount = self.picture_amount_slider.value()

        qt_signals.processing_module_request.emit(AugmentCoinCatalogRequest(
            catalog_path=catalog_dir_path,
            rotation=rotation,
            distortion=distortion,
            blur=blur,
            noise=noise,
            picture_amount=picture_amount
        ))

    def _handle_augmented_picture_response(self):
        pass

