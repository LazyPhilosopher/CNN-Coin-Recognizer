from core.qt_threading.headers.MessageBase import MessageBase


class CameraListMessage(MessageBase):
    def __init__(self, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
