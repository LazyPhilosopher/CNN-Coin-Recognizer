from PySide6.QtGui import QImage

from core.qt_threading.messages.MessageBase import MessageBase


class GrayscalePictureResponse(MessageBase):
    def __init__(self,  picture: str, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination


class ProcessedImageResponse(MessageBase):
    def __init__(self,  image: QImage, contour: list[tuple[int, int]], source=None, destination=None):
        super().__init__()
        self.image = image
        self.contour = contour
        self.source = source
        self.destination = destination


class BorderDetectionResponse(MessageBase):
    def __init__(self,  picture: str, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination
