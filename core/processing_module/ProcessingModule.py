from PySide6.QtCore import QObject, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import RequestBase, Modules
from core.qt_threading.headers.processing_module.GrayscalePictureRequest import GrayscalePictureRequest
from core.qt_threading.headers.processing_module.GrayscalePictureResponse import GrayscalePictureResponse

import cv2
import numpy as np


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
            # numpy_picture = self.QPixmapToArray(request.picture)
            # process_picture = self.process(numpy_picture)
            # response_picture = self.convertCvImage2QtImage(process_picture)

            self.qt_signals.processing_module_request.emit(GrayscalePictureResponse(
                source=Modules.PROCESSING_MODULE,
                destination=request.source,
                picture=self.convert_pixmap_to_grayscale(request.picture)
            ))

    def convert_pixmap_to_grayscale(self, pixmap: QPixmap):
        # Step 1: Convert QPixmap to QImage
        image = pixmap.toImage()

        # Step 2: Convert QImage to grayscale (QImage.Format_Grayscale8)
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)

        # Step 3: Convert the grayscale QImage back to QPixmap
        grayscale_pixmap = QPixmap.fromImage(grayscale_image)

        return grayscale_pixmap

    def process(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 2)
        img_canny = cv2.Canny(img_blur, 50, 9)
        img_dilate = cv2.dilate(img_canny, np.ones((4, 2)), iterations=11)
        img_erode = cv2.erode(img_dilate, np.ones((13, 7)), iterations=4)
        return cv2.bitwise_not(img_erode)

    def QPixmapToArray(self, pixmap):
        ## Get the size of the current pixmap
        size = pixmap.size()
        h = size.width()
        w = size.height()

        ## Get the QImage Item and convert it to a byte string
        qimg = pixmap.toImage()
        byte_str = qimg.bits().tobytes()

        ## Using the np.frombuffer function to convert the byte string into an np array
        img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))

        return img

    def convertCvImage2QtImage(self, data):
        height, width = data.shape
        qimage = QImage(data, width, height, QImage.Format_Grayscale8)

        return QPixmap.fromImage(qimage)
