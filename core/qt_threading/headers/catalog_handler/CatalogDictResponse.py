from core.qt_threading.headers.MessageBase import MessageBase


class CatalogDictResponse(MessageBase):
    def __init__(self, data: dict, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.catalog: dict = data
