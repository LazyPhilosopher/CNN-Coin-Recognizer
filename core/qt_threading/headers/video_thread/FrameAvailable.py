from PySide6.QtGui import QImage

from core.qt_threading.headers.MessageBase import MessageBase


class FrameAvailable(MessageBase):
    def __init__(self, frame: QImage, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.frame = frame
