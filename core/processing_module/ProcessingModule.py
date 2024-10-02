from PySide6.QtCore import QObject, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import RequestBase, Modules
from core.qt_threading.headers.processing_module.BorderDetectionRequest import BorderDetectionRequest


import cv2
import numpy as np

from core.qt_threading.headers.processing_module.BorderDetectionResponse import BorderDetectionResponse
from core.qt_threading.headers.processing_module.GrayscalePictureRequest import GrayscalePictureRequest
from core.qt_threading.headers.processing_module.GrayscalePictureResponse import GrayscalePictureResponse


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
            numpy_picture = self.QPixmapToArray(request.picture)
            process_picture = self.process(numpy_picture)
            response_picture = self.convertCvImage2QtImage(process_picture)

            self.qt_signals.processing_module_request.emit(GrayscalePictureResponse(
                source=Modules.PROCESSING_MODULE,
                destination=request.source,
                picture=self.convert_pixmap_to_grayscale(response_picture)
            ))

        elif isinstance(request, BorderDetectionRequest):
            picture = request.picture
            b_k = request.param_dict["b_k"]
            b_s = request.param_dict["b_s"]
            c_t1 = request.param_dict["c_t1"]
            c_t2 = request.param_dict["c_t2"]
            k1 = request.param_dict["k1"]
            k2 = request.param_dict["k2"]
            k3 = request.param_dict["k3"]
            k4 = request.param_dict["k4"]
            iter1 = request.param_dict["iter1"]
            iter2 = request.param_dict["iter2"]
            self.process(img=picture,
                         b_k=b_k,
                         b_s=b_s,
                         c_t1=c_t1,
                         c_t2=c_t2,
                         k1=k1,
                         k2=k2,
                         k3=k3,
                         k4=k4,
                         iter1=iter1,
                         iter2=iter2)
            self.qt_signals.processing_module_request.emit(BorderDetectionResponse(picture=picture))

    def convert_pixmap_to_grayscale(self, pixmap: QPixmap):
        # Step 1: Convert QPixmap to QImage
        image = pixmap.toImage()

        # Step 2: Convert QImage to grayscale (QImage.Format_Grayscale8)
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)

        # Step 3: Convert the grayscale QImage back to QPixmap
        grayscale_pixmap = QPixmap.fromImage(grayscale_image)

        return grayscale_pixmap

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

    def process(self, img, b_k, b_s, c_t1, c_t2, k1, k2, k3, k4, iter1, iter2):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b_k = b_k // 2 * 2 + 1
        img_blur = cv2.GaussianBlur(img_gray, (b_k, b_k), b_s)
        img_canny = cv2.Canny(img_blur, c_t1, c_t2)
        img_dilate = cv2.dilate(img_canny, np.ones((k1, k2)), iterations=iter1)
        img_erode = cv2.erode(img_dilate, np.ones((k3, k4)), iterations=iter2)
        return cv2.bitwise_not(img_erode)


