import os

from PySide6.QtCore import QTimer, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMainWindow, QApplication

from core.designer.pyqt6_designer.d_augmentation_window import Ui_AugmentationWindow
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import Modules, MessageBase
from core.qt_threading.messages.catalog_handler.Requests import CatalogDictRequest
from core.qt_threading.messages.catalog_handler.Responses import CatalogDictResponse
from core.qt_threading.messages.processing_module.Requests import AugmentedImageListRequest
from core.utilities.CaseInsensitiveDict import CaseInsensitiveDict
from core.utilities.helper import show_image_popup


class AugmentationWindow(QMainWindow, Ui_AugmentationWindow):
    def __init__(self):
        super().__init__()
        # self.is_running = False
        self.setupUi(self)
        self._window: QMainWindow | None = None
        self.coin_catalog: CaseInsensitiveDict = CaseInsensitiveDict()

        self.qt_signals = CommonSignals()
        # self.main_thread = QThread()

        self.generate_augmented_data_button.clicked.connect(self.handle_request_augmented_data_button)
        self.qt_signals.catalog_handler_response.connect(self.handle_request)
        QTimer.singleShot(0, self.request_coin_list)

    # def start_process(self):
    #     self.moveToThread(self.main_thread)
    #     self.main_thread.started.connect(self.worker)
    #     self.main_thread.start()
    #     self.is_running = True
    #
    # def worker(self):
    #     while self.is_running:
    #         QApplication.processEvents()

    def handle_request(self, request: MessageBase):
        request_handlers = {
            CatalogDictResponse: self.handle_catalog_dict_response
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def handle_catalog_dict_response(self, request: CatalogDictResponse):
        print(f"[AugmentationWindow]: {request}")
        self.coin_catalog = request.catalog

    def handle_request_augmented_data_button(self):
        for year, countries in self.coin_catalog.items():
            for country, coins in countries.items():
                for coin_name, coin in coins.items():
                    destination_dir = os.path.join("augmented_image_catalog", year, country, coin_name)
                    for picture, params_dict in coin.pictures.items():
                        destination_picture_dir = os.path.join(destination_dir, picture)
                        os.makedirs(destination_dir, exist_ok=True)

                        uncropped_image: QImage = QImage(os.path.join("coin_catalog", coin.coin_dir_path(), picture))
                        uncropped_image = uncropped_image.convertToFormat(QImage.Format_RGBA8888)
                        # show_image_popup(uncropped_image)

                        cropped_image = QImage(params_dict["cropped_version"])
                        cropped_image = cropped_image.convertToFormat(QImage.Format_RGBA8888)
                        # show_image_popup(cropped_image)

                        self.qt_signals.processing_module_request.emit(
                            AugmentedImageListRequest(uncropped_image=uncropped_image,
                                                      cropped_image=cropped_image,
                                                      destination_folder=destination_picture_dir,
                                                      source=Modules.IMAGE_COLLECTOR_WINDOW,
                                                      destination=Modules.PROCESSING_MODULE)
                        )

    def request_coin_list(self):
        self.qt_signals.catalog_handler_request.emit(CatalogDictRequest())


