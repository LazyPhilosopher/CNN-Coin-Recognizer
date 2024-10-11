import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ImageFrame(QFrame):
    def __init__(self, video_frame: QFrame):
        super().__init__(video_frame.parent())
        self.image_label = QLabel(self)
        self.image = QLabel(self)
        self.image_label.setScaledContents(True)
        self.contour_pixels: list[tuple[int, int]] = []
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.clone_properties_from(video_frame)
        video_frame.deleteLater()

    def clone_properties_from(self, other_frame):
        if isinstance(other_frame, QFrame):
            geometry = other_frame.geometry()
            frame_shape = other_frame.frameShape()
            frame_shadow = other_frame.frameShadow()
            self.setGeometry(geometry)
            self.setFrameShape(frame_shape)
            self.setFrameShadow(frame_shadow)

    def set_image(self, image: QImage):
        pixmap = QPixmap(image)
        self.image.setPixmap(pixmap)
        self.image_label.setPixmap(pixmap)

    def set_image_with_contour(self, image: QImage, contour: list[tuple[int, int]]):
        self.image.setPixmap(QPixmap(image))
        print(len(contour))
        self.set_contour_pixels(contour)
        image_without_background: QImage = self.remove_background(image, self.contour_pixels)
        self.set_image(image_without_background)

    def set_contour_pixels(self, contour: list[tuple[int, int]]):
        self.contour_pixels = contour

    def reset_contour_pixels(self):
        self.contour_pixels = []

    # def QPixmapToArray(self, pixmap):
    #     ## Get the size of the current pixmap
    #     size = pixmap.size()
    #     h = size.width()
    #     w = size.height()
    #
    #     ## Get the QImage Item and convert it to a byte string
    #     qimg = pixmap.toImage()
    #     byte_str = qimg.bits().tobytes()
    #
    #     ## Using the np.frombuffer function to convert the byte string into an np array
    #     img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))
    #
    #     return img
    def qimage_to_cv2(self, qimage):
        width = qimage.width()
        height = qimage.height()

        # Get the pointer to the data and reshape directly
        ptr = qimage.bits()
        arr = np.array(ptr).reshape((height, width, 3))  # Assuming 4 channels (RGBA)

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    def remove_background(self, img: QImage, contours):
        # Create a mask for the background
        image = self.qimage_to_cv2(img)
        mask = np.zeros_like(image[:, :, 0])

        if contours:
            cnt = max(contours, key=cv2.contourArea)  # Find the largest contour
            cv2.drawContours(mask, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)  # Fill the contour

        # Create a 3-channel version of the mask
        mask_rgb = cv2.merge([mask, mask, mask])

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, mask_rgb)

        # Set background to black or transparent (optional if saving as PNG with alpha)
        background = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        background[background == 255] = 0

        image = self.cv2_to_qimage(result)

        return image

    def cv2_to_qimage(self, cv_img):
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
        return qimage