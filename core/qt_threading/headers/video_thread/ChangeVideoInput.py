from core.qt_threading.headers.RequestBase import RequestBase

body: dict = {"device_id": None}


class ChangeVideoInput(RequestBase):
    def __init__(self, device_id: int, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.body = body
        self.body["device_id"] = device_id
