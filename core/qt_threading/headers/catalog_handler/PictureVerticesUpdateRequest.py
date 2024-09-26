from core.catalog.Coin import Coin
from core.qt_threading.headers.RequestBase import RequestBase


class PictureVerticesUpdateRequest(RequestBase):
    def __init__(self,  vertices, source=None, destination=None):
        super().__init__()
        self.vertices = vertices
        self.source = source
        self.destination = destination
