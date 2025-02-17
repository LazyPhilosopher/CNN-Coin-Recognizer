import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from core.gui.ImageCollector import ImageCollector
from core.modules.processing_module.ProcessingModule import ProcessingModule
from core.modules.video_module.video_module import VideoModule
from core.utilities.helper import resource_path, show_popup


class ImageCollectorApp:
    def __init__(self) -> None:
        app = QApplication(sys.argv)

        video_stream = VideoModule()
        video_stream.start_process()

        processing_module = ProcessingModule()
        processing_module.start_process()

        self.image_collector = ImageCollector()
        self.image_collector.show()

        sys.exit(app.exec())
