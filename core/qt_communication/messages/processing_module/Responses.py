from PySide6.QtGui import QImage

from core.qt_communication.base import MessageBase


class ProcessedImageResponse(MessageBase):
    def __init__(self,  image: QImage, source=None, destination=None):
        super().__init__()
        self.image = image
        self.source = source
        self.destination = destination


class AugmentedPictureResponse(MessageBase):
    def __init__(self,  augmented_image: QImage, augmented_hue: QImage, source=None, destination=None):
        super().__init__()
        self.augmented_image = augmented_image
        self.augmented_hue = augmented_hue
        self.source = source
        self.destination = destination
