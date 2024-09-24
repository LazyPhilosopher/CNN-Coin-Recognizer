from core.qt_threading.headers.RequestBase import RequestBase


class CatalogDictResponse(RequestBase):
    def __init__(self, data: dict, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.catalog: dict = data
