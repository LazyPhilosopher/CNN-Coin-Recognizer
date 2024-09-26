from enum import Enum

request_base: dict = {"header": {}, "body": {}}


class Modules(Enum):
    MAIN = 1
    VIDEO_STREAM = 2
    CATALOG_HANDLER = 3
    GALLERY_WINDOW = 4
    DRAGGABLE_CROSS_OVERLAY = 5


class RequestBase:
    def __init__(self):
        # Initialize the instance variables based on the dictionary keys
        # for key, value in request_base.items():
        #     setattr(self, key, value)
        self.source: Modules | None = None
        self.destination: Modules | None = None

        self.header = ""
        self.body = {}
