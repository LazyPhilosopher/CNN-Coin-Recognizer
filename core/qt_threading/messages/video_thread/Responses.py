from core.qt_threading.messages.MessageBase import MessageBase


class CameraListResponse(MessageBase):
    def __init__(self, camera_list: list, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.cameras = camera_list
