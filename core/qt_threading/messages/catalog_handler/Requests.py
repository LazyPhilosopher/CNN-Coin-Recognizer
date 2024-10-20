from PySide6.QtCore import QPoint
from PySide6.QtGui import QPixmap, QImage

from core.modules.catalog.Coin import Coin
from core.modules.catalog.ContourDetectionSettings import ContourDetectionSettings
from core.qt_threading.messages.MessageBase import MessageBase


class CatalogDictRequest(MessageBase):
    def __init__(self, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination


class PictureVerticesRequest(MessageBase):
    def __init__(self, coin: Coin, picture_filename: str, source=None, destination=None):
        super().__init__()
        self.source = source
        self.destination = destination
        self.coin = coin
        self.picture_filename = picture_filename


class SavePictureRequest(MessageBase):
    def __init__(self,
                 coin: Coin,
                 image_with_background: QImage,
                 cropped_image: QImage | None = None,
                 source=None,
                 destination=None):
        super().__init__()
        self.coin = coin
        self.image_with_background = image_with_background
        self.cropped_image = cropped_image
        self.source = source
        self.destination = destination


class SaveCroppedPictureRequest(MessageBase):
    def __init__(self,
                 coin: Coin,
                 picture_name: str,
                 image_without_background: QImage,
                 source=None,
                 destination=None):
        super().__init__()
        self.coin = coin
        self.picture_name = picture_name
        self.image_without_background = image_without_background
        self.source = source
        self.destination = destination


# class SaveVerticeCropPictureRequest(MessageBase):
#     def __init__(self,
#                  coin: Coin,
#                  picture_name: str,
#                  qpoint_list: list[QPoint],
#                  source=None,
#                  destination=None):
#         super().__init__()
#         self.coin = coin
#         self.picture_name = picture_name
#         self.qpoint_list = qpoint_list
#         self.source = source
#         self.destination = destination


class PictureVerticesUpdateRequest(MessageBase):
    def __init__(self,
                 vertices: list[tuple[int, int]],
                 coin: Coin,
                 picture_file: str,
                 source=None,
                 destination=None):
        super().__init__()
        self.vertices = vertices
        self.coin = coin
        self.picture_file = picture_file
        self.source = source
        self.destination = destination


class DeleteCroppedPicture(MessageBase):
    def __init__(self,
                 coin: Coin,
                 picture_file: str,
                 source=None,
                 destination=None):
        super().__init__()
        self.coin = coin
        self.picture_file = picture_file
        self.source = source
        self.destination = destination


class PictureRequest(MessageBase):
    def __init__(self,  coin: Coin, picture: str, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.picture = picture
        self.source = source
        self.destination = destination


class CroppedPictureRequest(MessageBase):
    def __init__(self,  coin: Coin, picture: str, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.picture = picture
        self.source = source
        self.destination = destination


class NewCoinRequest(MessageBase):
    def __init__(self,  coin_year: str, coin_country: str, coin_name: str, source=None, destination=None):
        super().__init__()
        self.coin_year = coin_year
        self.coin_country = coin_country
        self.coin_name = coin_name
        self.source = source
        self.destination = destination


class RemoveCoinRequest(MessageBase):
    def __init__(self,  coin: Coin, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.source = source
        self.destination = destination


class UpdateCoinCameraSettingsRequest(MessageBase):
    def __init__(self,  coin: Coin, params: ContourDetectionSettings, source=None, destination=None):
        super().__init__()
        self.coin = coin
        self.params: ContourDetectionSettings = params
        self.source = source
        self.destination = destination
