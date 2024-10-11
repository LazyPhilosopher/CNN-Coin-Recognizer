from PySide6.QtCore import QObject, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import MessageBase, Modules


import cv2
import numpy as np

from core.qt_threading.messages.processing_module.RemoveBackgroundDictionary import RemoveBackgroundDictionary
from core.qt_threading.messages.processing_module.Requests import GrayscalePictureRequest, DoNothingRequest, \
    RemoveBackgroundRequest
from core.qt_threading.messages.processing_module.Responses import ProcessedImageResponse


class ProcessingModule(QObject):

    def __init__(self):
        super().__init__()

        self.is_running = False
        self.main_thread = QThread()
        self.qt_signals = CommonSignals()
        self.qt_signals.processing_module_request.connect(self.handle_request)

    def start_process(self):
        self.moveToThread(self.main_thread)
        self.main_thread.started.connect(self.worker)
        self.main_thread.start()
        self.is_running = True

    def worker(self):
        while self.is_running:
            QApplication.processEvents()

    def handle_request(self, request: MessageBase):
        request_handlers = {
            GrayscalePictureRequest: self.handle_grayscale_picture,
            DoNothingRequest: self.handle_do_nothing,
            RemoveBackgroundRequest: self.handle_remove_background
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def handle_grayscale_picture(self, request: GrayscalePictureRequest):
        grayscale_image = request.image.convertToFormat(QImage.Format_Grayscale8)

        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=grayscale_image, source=Modules.PROCESSING_MODULE, destination=request.source))

    def handle_do_nothing(self, request: DoNothingRequest):
        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=request.image, source=Modules.PROCESSING_MODULE, destination=request.source))

    def handle_remove_background(self, request: RemoveBackgroundRequest):
        image = self.qimage_to_cv2(request.picture)
        params: dict = request.param_dict

        processed = self.process(image, params)
        contour, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(contour)

        img_no_bg = self.remove_background(image, contour)
        qimage_result = self.cv2_to_qimage(img_no_bg)

        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=request.picture, contour=contour, source=Modules.PROCESSING_MODULE, destination=request.source))

    def qimage_to_cv2(self, qimage):
        width = qimage.width()
        height = qimage.height()

        # Get the pointer to the data and reshape directly
        ptr = qimage.bits()
        arr = np.array(ptr).reshape((height, width, 3))  # Assuming 4 channels (RGBA)

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    def cv2_to_qimage(self, cv_img):
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
        return qimage

    def convert_pixmap_to_grayscale(self, pixmap: QPixmap):
        # Step 1: Convert QPixmap to QImage
        image = pixmap.toImage()

        # Step 2: Convert QImage to grayscale (QImage.Format_Grayscale8)
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)

        # Step 3: Convert the grayscale QImage back to QPixmap
        grayscale_pixmap = QPixmap.fromImage(grayscale_image)

        return grayscale_pixmap

    def QPixmapToNumpy(self, pixmap: QPixmap):
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

    def NumpyToQImage(self, data) -> QImage:
        height, width = data.shape
        return QImage(data, width, height, QImage.Format_Grayscale8)

    def process(self, img, params: RemoveBackgroundDictionary):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b_k = params["Blur Kernel"] // 2 * 2 + 1  # Ensure kernel size is odd
        img_blur = cv2.GaussianBlur(img_gray, (b_k, b_k), params["Blur Sigma"])
        img_canny = cv2.Canny(img_blur, params["Canny Threshold 1"], params["Canny Threshold 2"])
        img_dilate = cv2.dilate(img_canny, np.ones((params["Dilate Kernel1"], params["Dilate Kernel2"])), iterations=params["Dilate Iterations"])
        img_erode = cv2.erode(img_dilate, np.ones((params["Erode Kernel1"], params["Erode Kernel1"])), iterations=params["Erode Iterations"])
        return img_erode

    def remove_background(self, img: QImage, contours):
        # Create a mask for the background
        mask = np.zeros_like(img[:, :, 0])

        if contours:
            cnt = max(contours, key=cv2.contourArea)  # Find the largest contour
            cv2.drawContours(mask, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)  # Fill the contour

        # Create a 3-channel version of the mask
        mask_rgb = cv2.merge([mask, mask, mask])

        # Apply the mask to the original image
        result = cv2.bitwise_and(img, mask_rgb)

        # Set background to black or transparent (optional if saving as PNG with alpha)
        background = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        background[background == 255] = 0

        return result