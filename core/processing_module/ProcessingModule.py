import cv2
from queue import Queue
from PySide6.QtCore import QProcess, QObject, Slot, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import RequestBase, Modules
from core.qt_threading.headers.processing_module.GrayscalePictureRequest import GrayscalePictureRequest
from core.qt_threading.headers.processing_module.GrayscalePictureResponse import GrayscalePictureResponse
from core.qt_threading.headers.video_thread.CameraListRequest import CameraListRequest
from core.qt_threading.headers.video_thread.CameraListResponse import CameraListResponse
from core.qt_threading.headers.video_thread.ChangeVideoInput import ChangeVideoInput
from core.qt_threading.headers.video_thread.FrameAvailable import FrameAvailable


class ProcessingModule(QObject):

    def __init__(self):
        super().__init__()

        self.is_running = False
        self.main_thread = QThread()
        self.qt_signals = CommonSignals()
        self.qt_signals.processing_module_request.connect(self.receive_request)

    def start_process(self):
        self.moveToThread(self.main_thread)
        self.main_thread.started.connect(self.worker)
        self.main_thread.start()
        self.is_running = True

    def worker(self):
        while self.is_running:
            QApplication.processEvents()

    def receive_request(self, request: RequestBase):
        if isinstance(request, GrayscalePictureRequest):
            print(f"[ProcessingModule]: got request: {request}")
            self.qt_signals.processing_module_request.emit(GrayscalePictureResponse(
                source=Modules.PROCESSING_MODULE,
                destination=request.source,
                picture=self.convert_pixmap_to_grayscale(pixmap=request.picture)
            ))

    def convert_pixmap_to_grayscale(self, pixmap: QPixmap):
        # Step 1: Convert QPixmap to QImage
        image = pixmap.toImage()

        # Step 2: Convert QImage to grayscale (QImage.Format_Grayscale8)
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)

        # Step 3: Convert the grayscale QImage back to QPixmap
        grayscale_pixmap = QPixmap.fromImage(grayscale_image)

        return grayscale_pixmap
