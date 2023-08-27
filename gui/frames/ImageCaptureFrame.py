import math
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainterPath, QPen, QPixmap, QImage
from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtWidgets import (
    QMainWindow,
    QGraphicsView,
    QWidget,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QGraphicsScene,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QComboBox,
    QPushButton
)
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from utils.video import VideoStream

class NIPGraphicScene(QGraphicsScene):

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor(0x0, 0x0, 0x0))
        painter.setPen(QColor(0x33, 0x33, 0x33))
        painter.setBrush(QColor("black"))

        length = min(rect.width(), rect.height()) / 30

        x_left = QPointF(rect.left(), 0)
        x_right = QPointF(rect.right(), 0)
        painter.drawLine(x_left, x_right)

        right_triangle = QPainterPath()
        right_triangle.lineTo(-0.5 * math.sqrt(3) * length, 0.5 * length)
        right_triangle.lineTo(-0.5 * math.sqrt(3) * length, -0.5 * length)
        right_triangle.closeSubpath()
        right_triangle.translate(x_right)

        painter.drawPath(right_triangle)

        y_top = QPointF(0, rect.top())
        y_bottom = QPointF(0, rect.bottom())
        painter.drawLine(y_top, y_bottom)

        top_triangle = QPainterPath()
        top_triangle.lineTo(.5*length, -0.5 * math.sqrt(3) * length)
        top_triangle.lineTo(-.5*length, -0.5 * math.sqrt(3) * length)
        top_triangle.closeSubpath()
        top_triangle.translate(y_bottom)

        painter.drawPath(top_triangle)


class ImageCaptureFrame(QMainWindow):
    def __init__(self, parent=None, window_name=None):
        super().__init__()

        self.setWindowTitle("QGraphicsView Example")
        self.setGeometry(0, 0, 600, 400)  # Set a larger initial window size

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)

        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()

        self.graphics_view.setSceneRect(0, 0, 250, 250)  # Limit scene size to 250x250
        
        # Load an image using QPixmap
        pixmap = QPixmap("img\\Debugempty.png")  # Replace with the actual image path
        pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation) 
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)

        self.scene.addRect(QRectF(0, 0, 250, 250))
        self.graphics_view.setScene(self.scene)

        layout.addWidget(self.graphics_view, 0, 1, Qt.AlignTop | Qt.AlignLeft)  # Align in upper-left corner
        
        side_layout = QVBoxLayout()

        # Dropdown Menu
        self.dropdown = QComboBox()
        self.dropdown.addItem("Option 1")
        self.dropdown.addItem("Option 2")
        side_layout.addWidget(self.dropdown)

        # Button
        self.button = QPushButton("Click Me")
        side_layout.addWidget(self.button)

        layout.addLayout(side_layout, 0, 1, Qt.AlignTop | Qt.AlignCenter)  # Align in upper-right corner

    def set_picture(self, picture):
        # picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        height, width, channel = picture.shape
        bytes_per_line = 3 * width
        qimage = QImage(picture.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # create qpixmap from qimage
        pixmap = QPixmap.fromImage(qimage)
        # pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation) 
        # pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setPixmap(pixmap) 
        self.scene.update()
            



