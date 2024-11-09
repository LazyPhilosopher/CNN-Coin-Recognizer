from PySide6.QtCore import QPoint
from PySide6.QtGui import QPixmap, QImage

# from core.modules.catalog.ContourDetectionSettings import ContourDetectionSettings
from core.qt_communication.base import MessageBase


# class DoNothingRequest(MessageBase):
#     def __init__(self,  image: QImage, source=None, destination=None):
#         super().__init__()
#         self.image = image
#         self.source = source
#         self.destination = destination


# class GrayscalePictureRequest(MessageBase):
#     def __init__(self,  image: QImage, source=None, destination=None):
#         super().__init__()
#         self.image = image
#         self.source = source
#         self.destination = destination


# class BorderDetectionRequest(MessageBase):
#     def __init__(self,  picture: QImage, param_dict: dict, source=None, destination=None):
#         super().__init__()
#         self.picture = picture
#         self.param_dict = param_dict
#         self.source = source
#         self.destination = destination


class RemoveBackgroundRequest(MessageBase):
    def __init__(self,  picture: QImage, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.source = source
        self.destination = destination


# class RemoveBackgroundVerticesRequest(MessageBase):
#     def __init__(self,  picture: QImage, qpoint_vertices: list[QPoint], source=None, destination=None):
#         super().__init__()
#         self.picture = picture
#         self.qpoint_vertices = qpoint_vertices
#         self.source = source
#         self.destination = destination


# class AugmentedImageListRequest(MessageBase):
#     def __init__(self, uncropped_image: QImage, cropped_image: QImage, destination_folder: str, source=None, destination=None):
#         super().__init__()
#         self.uncropped_image = uncropped_image
#         self.cropped_image = cropped_image
#         self.destination_folder = destination_folder
#         self.source = source
#         self.destination = destination


class AugmentCoinCatalogRequest(MessageBase):
    def __init__(self,
                 catalog_path: str,
                 rotation: float,
                 distortion: float,
                 blur: float,
                 noise: float,
                 picture_amount: int,
                 source=None,
                 destination=None):
        super().__init__()
        self.catalog_path = catalog_path
        self.rotation = rotation
        self.distortion = distortion
        self.blur = blur
        self.noise = noise
        self.picture_amount = picture_amount
        self.source = source
        self.destination = destination
