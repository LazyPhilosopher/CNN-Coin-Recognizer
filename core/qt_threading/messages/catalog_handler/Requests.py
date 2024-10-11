from PySide6.QtGui import QPixmap, QImage

from core.modules.catalog.Coin import Coin
from core.qt_threading.messages.MessageBase import MessageBase


class CatalogDictRequest(MessageBase):
    def __init__(self, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination


class PictureContourRequest(MessageBase):
    def __init__(self, coin: Coin, picture_filename: str, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.coin = coin
        self.picture_filename = picture_filename


class SavePictureRequest(MessageBase):
    def __init__(self,  coin: Coin, image: QImage, contour: list[tuple[int, int]], source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.image = image
        self.contour = contour
        self.source = source
        self.destination = destination


class PictureContourUpdateRequest(MessageBase):
    def __init__(self,
                 contour: list[tuple[int, int]],
                 coin: Coin,
                 picture_file: str,
                 source=None,
                 destination=None):
        super().__init__()
        self.contour = contour
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


class NewCoinRequest(MessageBase):
    def __init__(self,  coin_year: str, coin_country: str, coin_name: str, source=None, destination=None):
        super().__init__()
        self.coin_year = coin_year
        self.coin_country = coin_country
        self.coin_name = coin_name
        self.source = source
        self.destination = destination


class RemoveCoinRequest(MessageBase):
    def __init__(self,  coin: Coin, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.source = source
        self.destination = destination
