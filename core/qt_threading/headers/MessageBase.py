from enum import Enum


class Modules(Enum):
    MAIN = 1
    VIDEO_STREAM = 2
    CATALOG_HANDLER = 3
    GALLERY_WINDOW = 4
    DRAGGABLE_CROSS_OVERLAY = 5
    PROCESSING_MODULE = 6
    ADD_NEW_PICTURE_WINDOW = 7


class MessageBase:
    def __init__(self):
        self.source: Modules | None = None
        self.destination: Modules | None = None
