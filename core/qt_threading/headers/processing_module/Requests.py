from PySide6.QtGui import QPixmap, QImage

from core.qt_threading.headers.MessageBase import MessageBase


class DoNothingRequest(MessageBase):
    def __init__(self,  image: QImage, source=None, destination=None):
        super().__init__()
        self.image = image
        self.source = source
        self.destination = destination


class GrayscalePictureRequest(MessageBase):
    def __init__(self,  image: QImage, source=None, destination=None):
        super().__init__()
        self.image = image
        self.source = source
        self.destination = destination


class BorderDetectionMessage(MessageBase):
    def __init__(self,  picture: str, param_dict: dict, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.param_dict = param_dict
        self.source = source
        self.destination = destination
