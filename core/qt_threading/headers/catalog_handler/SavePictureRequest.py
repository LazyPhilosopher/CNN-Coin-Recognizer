from PySide6.QtGui import QPixmap

from core.catalog.Coin import Coin
from core.qt_threading.headers.RequestBase import RequestBase


class SavePictureRequest(RequestBase):
    def __init__(self,  coin: Coin, picture: QPixmap, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.picture = picture
        self.source = source
        self.destination = destination
