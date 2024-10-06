from PySide6.QtCore import QPoint

from core.catalog.Coin import Coin
from core.qt_threading.headers.MessageBase import MessageBase


class PictureVerticesUpdateMessage(MessageBase):
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
