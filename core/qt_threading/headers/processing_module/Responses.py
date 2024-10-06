from core.qt_threading.headers.MessageBase import MessageBase


class GrayscalePictureResponse(MessageBase):
    def __init__(self,  picture: str, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination


class ProcessedPictureResponse(MessageBase):
    def __init__(self,  picture: str, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination


class BorderDetectionResponse(MessageBase):
    def __init__(self,  picture: str, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination
