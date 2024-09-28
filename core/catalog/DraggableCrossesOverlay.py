from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QPicture
from PySide6.QtWidgets import QWidget

from core.catalog import Coin

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import Modules
from core.qt_threading.headers.catalog_handler.PictureVerticesUpdateRequest import PictureVerticesUpdateRequest


class DraggableCrossesOverlay(QWidget):
    mouse_released = Signal(object)

    def __init__(self, parent=None, ):
        super().__init__(parent)
        self.crosses: list[QPoint] = []
        self.signals = CommonSignals()
        self.selected_cross = None
        self.cross_size = 10
        self.drawn_coin: Coin = None

        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)  # Optional, for smoother dragging

    # def init_image_with_vertices(self, coin: Coin, ):
    #
    #     self.drawn_coin = coin
    #
    #     width: int = coin_picture.width()
    #     height: int = coin_picture.height()
    #     crosses = [QPoint(x * width, y * height) for (x, y) in vertices]

    def paintEvent(self, event):
        if not self.parent():
            return

        painter = QPainter(self)
        pen = QPen(QColor('red'), 2)
        painter.setPen(pen)
        self.draw_crosses(painter=painter)

    def draw_crosses(self, painter: QPainter):
        if len(self.crosses) > 2:
            drawn_crosses = [*self.crosses, self.crosses[0]]
            for i in range(0, len(self.crosses), 1):
                a: QPoint = drawn_crosses[i]
                b: QPoint = drawn_crosses[i+1]
                self.draw_cross(painter, a)
                self.draw_cross(painter, b)
                painter.drawLine(a, b)
        else:
            for i in range(0, len(self.crosses), 1):
                a: QPoint = self.crosses[i]
                self.draw_cross(painter, a)

    def draw_cross(self, painter, center):
        painter.drawLine(center.x() - self.cross_size, center.y(),
                         center.x() + self.cross_size, center.y())
        painter.drawLine(center.x(), center.y() - self.cross_size,
                         center.x(), center.y() + self.cross_size)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.crosses.append(QPoint(event.pos().x(), event.pos().y()))
            self.update()
        else:
            for i, pos in enumerate(self.crosses):
                if (abs(pos.x() - event.pos().x()) < self.cross_size and
                        abs(pos.y() - event.pos().y()) < self.cross_size):
                    self.selected_cross = i
                    break

    def mouseMoveEvent(self, event):
        if self.selected_cross is not None:
            self.crosses[self.selected_cross] = event.pos()
            self.update()  # Redraw the widget

    def mouseReleaseEvent(self, event):
        self.selected_cross = None
        self.mouse_released.emit(self.crosses)

    def reset_vertices(self):
        self.crosses = []
        self.update()
