from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMainWindow

from core.designer.pyqt6_designer.d_augmentation_window import Ui_AugmentationWindow
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import Modules
from core.qt_threading.messages.processing_module.Requests import AugmentedImageListRequest


class AugmentationWindow(QMainWindow, Ui_AugmentationWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._window: QMainWindow | None = None

        self.qt_signals = CommonSignals()

        self.generate_augmented_data_button.clicked.connect(self.handle_request_augmented_data_button)

    def handle_request_augmented_data_button(self):
        image = QImage('D:\\Projects\\bachelor_thesis\\OpenCV2-Coin-Recognizer\\coin_catalog\\cropped\\2018\\Czech Republic\\1 Koruna\\f593ab1d-a271-415f-a566-7039e291cd22.png')
        self.qt_signals.processing_module_request.emit(
            AugmentedImageListRequest(image=image, source=Modules.IMAGE_COLLECTOR_WINDOW, destination=Modules.PROCESSING_MODULE)
        )

