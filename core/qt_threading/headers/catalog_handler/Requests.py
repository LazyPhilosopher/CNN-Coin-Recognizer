from PySide6.QtGui import QPixmap

from core.catalog.Coin import Coin
from core.qt_threading.headers.MessageBase import MessageBase


class CatalogDictRequest(MessageBase):
    def __init__(self, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination


class PictureVerticesRequest(MessageBase):
    def __init__(self, coin: Coin, picture_filename: str, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.coin = coin
        self.picture_filename = picture_filename


class SavePictureRequest(MessageBase):
    def __init__(self,  coin: Coin, picture: QPixmap, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.picture = picture
        self.source = source
        self.destination = destination


class PictureVerticesUpdateRequest(MessageBase):
    def __init__(self,
                 vertices: list[tuple[int, int]],
                 coin: Coin,
                 picture_file: str,
                 source=None,
                 destination=None):
        super().__init__()
        self.vertices = vertices
        self.coin = coin
        self.picture_file = picture_file
        self.source = source
        self.destination = destination


class PictureRequest(MessageBase):
    def __init__(self,  coin: Coin, picture: str, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.picture = picture
        self.source = source
        self.destination = destination
