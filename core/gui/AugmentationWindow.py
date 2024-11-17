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

        self.sliders = [
            self.blur_slider,
            self.noise_slider,
            self.rotation_slider,
            self.distortion_slider,
            self.picture_amount_slider
        ]
        [slider.valueChanged.connect(self.update_labels) for slider in self.sliders]

        # self.labels = [
        #     self.blur_slider,
        #     self.noise_slider,
        #     self.rotation_slider,
        #     self.distortion_slider,
        #     self.picture_amount_slider
        # ]
        # [label.valueChanged.connect(self.update_sliders) for label in self.labels]

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

    def update_labels(self):
        [slider.valueChanged.disconnect(self.update_labels) for slider in self.sliders]

        self.blur_label.setText(str(self.blur_slider.value()))
        self.noise_label.setText(str(self.noise_slider.value()))
        self.rotation_label.setText(str(self.rotation_slider.value()))
        self.distorsion_label.setText(str(self.distortion_slider.value()))
        self.number_of_pictures_label.setText(str(self.picture_amount_slider.value()))

        [slider.valueChanged.connect(self.update_labels) for slider in self.sliders]

    # def update_sliders(self):
    #     [label.valueChanged.disconnect(self.update_labels) for label in self.labels]
    #
    #     self.blur_slider.setValue(int(self.blur_label.valu()))
    #     self.noise_slider.setValue(int(self.noise_label.value()))
    #     self.rotation_slider.setValue(int(self.rotation_label.value()))
    #     self.distortion_slider.setValue(int(self.distorsion_label.value()))
    #     self.picture_amount_slider.setValue(int(self.picture_amount_slider.value()))
    #
    #     [label.valueChanged.connect(self.update_labels) for label in self.labels]


    def _handle_augmented_picture_response(self):
        pass

