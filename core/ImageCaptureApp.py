import sys

from PySide6.QtWidgets import QApplication

from core.modules.catalog.CatalogController import CoinCatalogHandler
from core.modules.processing_module.ProcessingModule import ProcessingModule
from core.qt_threading.common_signals import CommonSignals
from core.designer.AppWindow import AppWindow
from core.modules.video.Video import VideoStream


class ImageCaptureApp:
    def __init__(self) -> None:
        app = QApplication([])
        self.signals = CommonSignals()

        video_stream = VideoStream()
        video_stream.start_process()

        catalog_handler = CoinCatalogHandler()
        catalog_handler.start_process()

        processing_module = ProcessingModule()
        processing_module.start_process()

        self.main_window = AppWindow()
        self.main_window.show()

        sys.exit(app.exec())

