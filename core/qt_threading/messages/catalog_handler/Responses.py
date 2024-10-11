from core.qt_threading.messages.MessageBase import MessageBase


class PictureContourResponse(MessageBase):
    def __init__(self, contour: list[tuple[int, int]], source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.contour = contour


class PictureResponse(MessageBase):
    def __init__(self,  picture, contour, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.contour = contour
        self.source = source
        self.destination = destination


class CatalogDictResponse(MessageBase):
    def __init__(self, data: dict, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.catalog: dict = data
