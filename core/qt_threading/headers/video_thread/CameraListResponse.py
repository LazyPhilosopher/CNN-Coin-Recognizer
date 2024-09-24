from core.qt_threading.headers.RequestBase import RequestBase

body: dict = {"cameras": {}}


class CameraListResponse(RequestBase):
    def __init__(self, camera_list: list, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.body = body
        self.body["cameras"] = camera_list
