from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ImageFrame(QFrame):
    def __init__(self, video_frame: QFrame):
        super().__init__(video_frame.parent())
        self.image_label = QLabel(self)
        layout = QVBoxLayout()
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

    def set_image(self, ndarray):
        if ndarray.ndim == 2:  # Grayscale image
            height, width = ndarray.shape
            bytes_per_line = width
            q_image = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif ndarray.ndim == 3:  # Color image
            height, width, channels = ndarray.shape
            if channels == 3:  # RGB image
                bytes_per_line = channels * width
                q_image = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_RGB888)
            elif channels == 4:  # RGBA image
                bytes_per_line = channels * width
                q_image = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            else:
                raise ValueError("Unsupported number of channels")
        else:
            raise ValueError("Unsupported ndarray shape")

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
