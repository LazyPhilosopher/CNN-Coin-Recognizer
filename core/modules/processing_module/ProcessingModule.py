from PySide6.QtCore import QObject, QThread, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from core.modules.catalog.ContourDetectionSettings import ContourDetectionSettings
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import MessageBase, Modules


import cv2
import numpy as np

from core.qt_threading.messages.processing_module.RemoveBackgroundDictionary import RemoveBackgroundSliderDictionary
from core.qt_threading.messages.processing_module.Requests import GrayscalePictureRequest, DoNothingRequest, \
    RemoveBackgroundRequest, RemoveBackgroundVerticesRequest
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
            RemoveBackgroundRequest: self.handle_remove_background,
            RemoveBackgroundVerticesRequest: self.handle_remove_background_vertices
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
        params: dict = request.params

        processed = self.process(image, params)
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(' '.join(str(*array[0]) for array in contours)+"\n")

        img_no_bg = self.remove_background(image, contours)
        qimage_result = self.cv2_to_qimage(img_no_bg)

        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=qimage_result, mask=contours, source=Modules.PROCESSING_MODULE, destination=request.source))

    def handle_remove_background_vertices(self, request: RemoveBackgroundVerticesRequest):
        # Convert QImage to OpenCV image
        image = self.qimage_to_cv2(request.picture)

        # Get the list of QPoint objects
        vertices: list[QPoint] = request.qpoint_vertices

        # Convert list of QPoint to a NumPy array
        points = np.array([[point.x(), point.y()] for point in vertices], dtype=np.int32)

        # Create a mask with the same size as the original image, filled with zeros (black)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Fill the mask with the polygon defined by the vertices
        cv2.fillPoly(mask, [points], 255)  # The polygon is filled with white (255)

        # Convert the image to BGRA format (4 channels) to include transparency
        img_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        # Set pixels outside the polygon to transparent (0 alpha)
        img_with_alpha[mask == 0] = [0, 0, 0, 0]  # Set to transparent (RGBA)

        # Convert the resulting image back to QImage
        qimage_result = self.cv2_to_qimage(img_with_alpha)

        # Emit the processed result
        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=qimage_result, mask=points, source=Modules.PROCESSING_MODULE,
                                   destination=request.source)
        )

    def qimage_to_cv2(self, qimage):
        width = qimage.width()
        height = qimage.height()

        # Check the format of the QImage to determine if it has an alpha channel
        if qimage.format() == QImage.Format_RGBA8888:
            channels = 4  # RGBA
        else:
            channels = 3  # RGB

        # Get the pointer to the data and reshape it based on the number of channels
        ptr = qimage.bits()
        arr = np.array(ptr).reshape((height, width, channels))

        if channels == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)  # Keep the alpha channel
        else:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    def cv2_to_qimage(self, cv_img):
        # Ensure that the image is contiguous in memory
        cv_img = np.ascontiguousarray(cv_img)

        # Extract height, width, and number of channels from the image
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width

        # Choose the appropriate QImage format based on the number of channels
        if channels == 4:  # BGRA to RGBA
            qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:
            qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

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

    def process(self, img, params: ContourDetectionSettings):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b_k = params.blur_kernel // 2 * 2 + 1  # Ensure kernel size is odd
        img_blur = cv2.GaussianBlur(img_gray, (b_k, b_k), params.blur_sigma)
        img_canny = cv2.Canny(img_blur, params.canny_threshold_1, params.canny_threshold_2)
        img_dilate = cv2.dilate(img_canny, np.ones((params.dilate_kernel_1, params.dilate_kernel_2)), iterations=params.dilate_iteration)
        img_erode = cv2.erode(img_dilate, np.ones((params.erode_kernel_1, params.erode_kernel_2)), iterations=params.erode_iteration)
        return img_erode

    def remove_background(self, img, contours):
        # Create a mask for the background (single channel)
        mask = np.zeros_like(img[:, :, 0])

        if contours:
            cnt = max(contours, key=cv2.contourArea)  # Find the largest contour
            cv2.drawContours(mask, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)  # Fill the contour

        # Convert the mask to a 4-channel version (for RGBA)
        mask_alpha = cv2.merge([mask, mask, mask, mask])

        # Create a transparent image
        img_with_alpha = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)

        # Set background pixels to transparent where the mask is zero
        img_with_alpha[mask_alpha[:, :, 3] == 0] = [0, 0, 0, 0]  # Set to transparent (RGBA)

        return img_with_alpha
