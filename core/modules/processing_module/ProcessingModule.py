import os
import uuid

from PySide6.QtCore import QObject, QThread, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

import imgaug.augmenters as iaa

from core.modules.catalog.Coin import Coin
from core.modules.catalog.ContourDetectionSettings import ContourDetectionSettings
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import MessageBase, Modules


import cv2
import numpy as np

from core.qt_threading.messages.processing_module.RemoveBackgroundDictionary import RemoveBackgroundSliderDictionary
from core.qt_threading.messages.processing_module.Requests import GrayscalePictureRequest, DoNothingRequest, \
    RemoveBackgroundRequest, RemoveBackgroundVerticesRequest, AugmentedImageListRequest
from core.qt_threading.messages.processing_module.Responses import ProcessedImageResponse
from core.utilities.helper import cv2_to_qimage, qimage_to_cv2, remove_background, transparent_to_hue, show_image_popup


class ProcessingModule(QObject):

    def __init__(self):
        super().__init__()

        self.is_running = False
        self.is_processing = False
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
            RemoveBackgroundVerticesRequest: self.handle_remove_background_vertices,
            AugmentedImageListRequest: self.handle_augmented_image_list_request
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            if self.is_processing:
                return

            self.is_processing = True
            handler(request)
            self.is_processing = False

    def handle_grayscale_picture(self, request: GrayscalePictureRequest):
        grayscale_image = request.image.convertToFormat(QImage.Format_Grayscale8)

        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=grayscale_image, source=Modules.PROCESSING_MODULE, destination=request.source))

    def handle_do_nothing(self, request: DoNothingRequest):
        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=request.image, source=Modules.PROCESSING_MODULE, destination=request.source))

    def handle_remove_background(self, request: RemoveBackgroundRequest):
        image = qimage_to_cv2(request.picture)
        params: dict = request.params

        processed = self.process(image, params)
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(' '.join(str(*array[0]) for array in contours)+"\n")

        img_no_bg = remove_background(image, contours)
        qimage_result = cv2_to_qimage(img_no_bg)

        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=qimage_result, mask=contours, source=Modules.PROCESSING_MODULE, destination=request.source))

    def handle_remove_background_vertices(self, request: RemoveBackgroundVerticesRequest):
        # Convert QImage to OpenCV image
        image = qimage_to_cv2(request.picture)

        # Get the list of QPoint objects
        vertices: list[QPoint] = request.qpoint_vertices

        # Convert list of QPoint to a NumPy array
        points = np.array([[point.x(), point.y()] for point in vertices], dtype=np.int32)

        # Create a mask with the same size as the original image, filled with zeros (black)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Fill the mask with the polygon defined by the vertices
        cv2.fillPoly(mask, [points], 255)  # The polygon is filled with white (255)

        # Convert the image to BGRA format (4 channels) to include transparency
        img_with_alpha = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        # Set pixels outside the polygon to transparent (0 alpha)
        img_with_alpha[mask == 0] = [0, 0, 0, 0]  # Set to transparent (RGBA)

        # Convert the resulting image back to QImage
        qimage_result = cv2_to_qimage(img_with_alpha)

        # Emit the processed result
        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=qimage_result, mask=points, source=Modules.PROCESSING_MODULE,
                                   destination=request.source)
        )

    def handle_augmented_image_list_request(self, request: AugmentedImageListRequest):
        uncropped_image_np = qimage_to_cv2(request.uncropped_image)
        cropped_image_np = qimage_to_cv2(request.cropped_image)

        os.makedirs(os.path.join(f"{request.destination_folder}\\uncropped-augmented"), exist_ok=True)
        os.makedirs(os.path.join(f"{request.destination_folder}\\cropped-augmented"), exist_ok=True)
        os.makedirs(os.path.join(f"{request.destination_folder}\\hue"), exist_ok=True)

        seq = iaa.Sequential([
            # Rotate randomly between -180 and +180 degrees
            iaa.Affine(rotate=(-180, 180)),
            # Apply local distortions with random scale between 0.001 and 0.005
            iaa.PiecewiseAffine(scale=(0.005, 0.02)),
            iaa.Affine(rotate=(-180, 180), scale=(0.25, 1.2)),  # Scale between 0.5x and 1.2x
            # Apply Gaussian blur with a random sigma between 0.4 and 0.5
            iaa.GaussianBlur(sigma=(0.1, 0.5))
        ])

        noise = iaa.Sequential([
            # Add Gaussian noise with random scale between 0 and 0.05*255
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
        ])

        # Create a deterministic augmentation from the sequence
        # deterministic_seq = seq.to_deterministic()
        # deterministic_noise = noise.to_deterministic()

        picture_name: str = uuid.uuid4()

        for i in range(10):

            full_picture_np = seq.augment_image(uncropped_image_np)
            full_picture_np = noise.augment_image(full_picture_np)
            full_image: QImage = cv2_to_qimage(full_picture_np)
            # full_image = full_image.convertToFormat(QImage.Format_RGB888)
            full_pixmap: QPixmap = QPixmap.fromImage(full_image)
            full_pixmap.save(f"{request.destination_folder}\\uncropped-augmented\\{picture_name}_{i}.png")

            cropped_picture_np = seq.augment_image(cropped_image_np)
            cropped_picture_np = noise.augment_image(cropped_picture_np)
            cropped_image: QImage = cv2_to_qimage(cropped_picture_np)
            # cropped_image = cropped_image.convertToFormat(QImage.Format_RGB888)
            cropped_pixmap: QPixmap = QPixmap.fromImage(cropped_image)
            cropped_pixmap.save(f"{request.destination_folder}\\cropped-augmented\\{picture_name}_{i}.png")

            hue_image_np = transparent_to_hue(cropped_image_np)
            augmented_hue_image_np = seq.augment_image(hue_image_np)
            augmented_hue_image: QImage = cv2_to_qimage(augmented_hue_image_np)
            # augmented_hue_image = augmented_hue_image.convertToFormat(QImage.Format_RGB888)
            augmented_hue_pixmap = QPixmap.fromImage(augmented_hue_image)
            augmented_hue_pixmap.save(f"{request.destination_folder}\\hue\\{picture_name}_{i}.png")

        print("done")


    def process(self, img, params: ContourDetectionSettings):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b_k = params.blur_kernel // 2 * 2 + 1  # Ensure kernel size is odd
        img_blur = cv2.GaussianBlur(img_gray, (b_k, b_k), params.blur_sigma)
        img_canny = cv2.Canny(img_blur, params.canny_threshold_1, params.canny_threshold_2)
        img_dilate = cv2.dilate(img_canny, np.ones((params.dilate_kernel_1, params.dilate_kernel_2)), iterations=params.dilate_iteration)
        img_erode = cv2.erode(img_dilate, np.ones((params.erode_kernel_1, params.erode_kernel_2)), iterations=params.erode_iteration)
        return img_erode


