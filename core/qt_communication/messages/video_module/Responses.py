from PIL.ImageQt import QImage

from core.qt_communication.base import MessageBase


class CameraListResponse(MessageBase):
    def __init__(self, camera_list: list, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.cameras = camera_list


class FrameAvailable(MessageBase):
    def __init__(self, frame: QImage, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.frame = frame
