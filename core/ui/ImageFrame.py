import numpy as np

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ImageFrame(QFrame):
    def __init__(self, video_frame: QFrame):
        super().__init__(video_frame.parent())
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
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

    def set_image(self, image_data):
        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 2:  # Grayscale image
                height, width = image_data.shape
                bytes_per_line = width
                q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif image_data.ndim == 3:  # Color image
                height, width, channels = image_data.shape
                if channels == 3:  # RGB image
                    bytes_per_line = channels * width
                    q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                elif channels == 4:  # RGBA image
                    bytes_per_line = channels * width
                    q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
                else:
                    raise ValueError("Unsupported number of channels")
            else:
                raise ValueError("Unsupported ndarray shape")

            pixmap = QPixmap.fromImage(q_image)
        else:
            pixmap = image_data

        self.image_label.setPixmap(pixmap)
        # self.image_label.setAlignment(Qt.AlignCenter)

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
