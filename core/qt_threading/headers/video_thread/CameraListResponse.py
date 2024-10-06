from core.qt_threading.headers.MessageBase import MessageBase

body: dict = {"cameras": {}}


class CameraListResponse(MessageBase):
    def __init__(self, camera_list: list, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.body = body
        self.body["cameras"] = camera_list
