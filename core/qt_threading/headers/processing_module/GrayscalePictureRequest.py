from PySide6.QtGui import QPixmap

from core.catalog.Coin import Coin
from core.qt_threading.headers.RequestBase import RequestBase


class GrayscalePictureRequest(RequestBase):
    def __init__(self,  picture: QPixmap, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination
