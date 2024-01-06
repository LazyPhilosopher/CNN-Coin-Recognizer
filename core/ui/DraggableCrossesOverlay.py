from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget


class DraggableCrossesOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.crosses = []
        self.selected_cross = None
        self.cross_size = 10

        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)  # Optional, for smoother dragging

    def paintEvent(self, event):
        if not self.parent():
            return

        painter = QPainter(self)
        pen = QPen(QColor('red'), 2)
        painter.setPen(pen)

        for pos in self.crosses:
            self.draw_cross(painter, pos)

    def draw_cross(self, painter, center):
        painter.drawLine(center.x() - self.cross_size, center.y(),
                         center.x() + self.cross_size, center.y())
        painter.drawLine(center.x(), center.y() - self.cross_size,
                         center.x(), center.y() + self.cross_size)

    def mousePressEvent(self, event):
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
