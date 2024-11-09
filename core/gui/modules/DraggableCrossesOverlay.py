from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

# from core.modules.catalog import Coin

# from core.qt_threading.common_signals import CommonSignals


class DraggableCrossesOverlay(QWidget):
    mouse_released = Signal(object)
    crosses_changed = Signal()

    def __init__(self, parent=None, ):
        super().__init__(parent)
        self.crosses: list[QPoint] = []
        # self.signals = CommonSignals()
        self.selected_cross = None
        self.cross_size = 10
        # self.drawn_coin: Coin = None

        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)  # Optional, for smoother dragging

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

    def reset_crosses(self):
        self.crosses = []
        self.crosses_changed.emit()

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
        self.crosses_changed.emit()

    # def reset_vertices(self):
    #     self.crosses = []
    #     self.update()
