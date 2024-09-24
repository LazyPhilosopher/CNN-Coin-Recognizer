import sys

from PySide6.QtWidgets import QApplication

from core.catalog.CatalogController import CoinCatalogHandler
from core.qt_threading.common_signals import CommonSignals
from core.ui.AppWindow import AppWindow
from core.video.video import VideoStream


class ImageCaptureApp:
    def __init__(self) -> None:
        app = QApplication([])
        self.signals = CommonSignals()

        video_stream = VideoStream()
        video_stream.start_process()

        catalog_handler = CoinCatalogHandler()
        catalog_handler.start_process()

        self.main_window = AppWindow()
        self.main_window.show()

        sys.exit(app.exec())

