from core.catalog.Coin import Coin
from core.qt_threading.headers.RequestBase import RequestBase


class BorderDetectionRequest(RequestBase):
    def __init__(self,  picture: str, param_dict: dict, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.param_dict = param_dict
        self.source = source
        self.destination = destination
