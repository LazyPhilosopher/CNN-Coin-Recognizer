from core.qt_threading.headers.RequestBase import RequestBase


class PictureResponse(RequestBase):
    def __init__(self,  picture, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination
