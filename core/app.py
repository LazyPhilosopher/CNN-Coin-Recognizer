import sys

from PySide6.QtWidgets import QApplication

from core.modules.processing.processing_module import ProcessingModule
from core.modules.video.video_module import VideoModule
from core.qt_communication.messages.common_signals import CommonSignals
from core.gui.AppWindow import AppWindow


class OpenCV2CoinRecognizerApp:
    def __init__(self) -> None:
        app = QApplication([])
        self.signals = CommonSignals()

        video_stream = VideoModule()
        video_stream.start_process()

        processing_module = ProcessingModule()
        processing_module.start_process()

        self.main_window = AppWindow()
        self.main_window.show()

        sys.exit(app.exec())

