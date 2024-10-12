from PySide6.QtGui import QImage

from core.qt_threading.messages.MessageBase import MessageBase


class PictureVerticesResponse(MessageBase):
    def __init__(self, vertices: list[tuple[int, int]], source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.vertices = vertices


class PictureResponse(MessageBase):
    def __init__(self,  pic_with_background: QImage, pic_no_background: QImage, source=None, destination=None):
        super().__init__()
        self.pic_with_background = pic_with_background
        self.pic_no_background = pic_no_background
        self.source = source
        self.destination = destination


class CatalogDictResponse(MessageBase):
    def __init__(self, data: dict, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.catalog: dict = data
