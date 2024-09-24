from core.qt_threading.headers.RequestBase import RequestBase


class FrameAvailable(RequestBase):
    def __init__(self, frame, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.frame = frame
