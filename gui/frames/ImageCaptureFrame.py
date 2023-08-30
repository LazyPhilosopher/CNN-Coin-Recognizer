from distutils.sysconfig import customize_compiler
from hmac import new
import math
# from PyQt5 import QtGui
# import cv2
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
    QPushButton,
    QLineEdit
)
from PyQt5.QtWidgets import QWidget, QVBoxLayout
# from utils.video import VideoStream

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


class SaveImageButton(QPushButton):
    def __init__(self,  custom_signals, pixmap_item):
        super(SaveImageButton, self).__init__(text="Save image")
        self.custom_signals = custom_signals
        self.pixmap_item = pixmap_item
        
    def mousePressEvent(self, e) -> None:
        super().mousePressEvent(e)
        self.pixmap_item.pixmap().save("test.jpg")
        # self.custom_signals.save_picture.emit(self.pixmap)
        # print("Save Button clicked!")

class NewCoinButton(QPushButton):
    def __init__(self,  custom_signals, pixmap_item):
        super(NewCoinButton, self).__init__(text="Create new coin")
        self.custom_signals = custom_signals
        self.pixmap_item = pixmap_item
        
    def mousePressEvent(self, e) -> None:
        super().mousePressEvent(e)
        self.pixmap_item.pixmap().save("test.jpg")
        # self.custom_signals.save_picture.emit(self.pixmap)
        # print("New Coin Button clicked!")

class ImageCaptureFrame(QMainWindow):
    def __init__(self, parent=None, custom_signals=None, list_camera_ids_list=None):
        super().__init__()
        self.custom_signals = custom_signals
        self.list_camera_ids_list = list_camera_ids_list
        self.setWindowTitle("QGraphicsView Example")
        self.setGeometry(0, 0, 600, 400)  # Set a larger initial window size

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)

        self.graphics_view = QGraphicsView()
        self.graphics_view.setSceneRect(0, 0, 250, 250)  # Limit scene size to 250x250
        layout.addWidget(self.graphics_view, 0, 1, Qt.AlignTop | Qt.AlignLeft)  # Align in upper-left corner
        
        self.scene = QGraphicsScene()
        self.scene.addRect(QRectF(0, 0, 250, 250))
        self.graphics_view.setScene(self.scene)
        
        pixmap = QPixmap("img\\Debugempty.png")
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.sized_pixmap = QGraphicsPixmapItem(pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.scene.addItem(self.sized_pixmap)
        
        # Side Layout setup
        side_layout = QVBoxLayout()

        layout.addLayout(side_layout, 0, 2, Qt.AlignTop | Qt.AlignCenter)  # Align in upper-right corner

        # Dropdown Menu
        self.dropdown = QComboBox()
        for camera in self.list_camera_ids_list:
            self.dropdown.addItem(f"Camera {camera}")
        self.dropdown.activated.connect(self.handleDropdownActivated)
        side_layout.addWidget(self.dropdown)
        
        # Coin name field
        self.coin_name_textbox = QLineEdit(self)
        self.coin_name_textbox.setFixedWidth(150)  # Set the desired width
        side_layout.addWidget(self.coin_name_textbox)

        # New coin button
        self.button = NewCoinButton(custom_signals=self.custom_signals, pixmap_item=self.pixmap_item)
        side_layout.addWidget(self.button)

        # Save image button
        self.button = SaveImageButton(custom_signals=self.custom_signals, pixmap_item=self.pixmap_item)
        side_layout.addWidget(self.button)

    def set_picture(self, picture):
        # picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        height, width, channel = picture.shape
        bytes_per_line = 3 * width
        qimage = QImage(picture.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # create qpixmap from qimage
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item.setPixmap(pixmap)
        self.sized_pixmap.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # pixmap_item = QGraphicsPixmapItem(pixmap)
        # self.sized_pixmap.setPixmap(self.sized_pixmap) 
        self.scene.update()
        
    def handleDropdownActivated(self, index):
        self.custom_signals.camera_reinit_signal.emit(index)
            