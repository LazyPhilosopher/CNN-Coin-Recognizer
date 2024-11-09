from PySide6.QtGui import QImage

from core.qt_communication.base import MessageBase


# class NewPhotoMessage(MessageBase):
#     def __init__(self):
#         super().__init__()


# class FrameAvailable(MessageBase):
#     def __init__(self, frame: QImage, source=None, destination=None):
#         super().__init__()
#         self.source = source
#         self.destination = destination
#         self.frame = frame


# class ChangeVideoInput(MessageBase):
#     def __init__(self, device_id: int, source=None, destination=None):
#         super().__init__()
#         self.source = source
#         self.destination = destination
#         self.device_id = device_id


class CameraListRequest(MessageBase):
    def __init__(self, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
