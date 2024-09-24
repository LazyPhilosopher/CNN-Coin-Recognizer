from core.qt_threading.headers.RequestBase import RequestBase


class CameraListRequest(RequestBase):
    def __init__(self, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
