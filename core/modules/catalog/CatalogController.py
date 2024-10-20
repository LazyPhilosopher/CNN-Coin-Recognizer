import json
import os
import uuid

from PySide6.QtCore import QObject, QThread
from PySide6.QtGui import QPixmap, QImage

from core.modules.catalog.Coin import Coin
from core.modules.catalog.ContourDetectionSettings import ContourDetectionSettings
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import MessageBase, Modules
from core.qt_threading.messages.catalog_handler.Requests import PictureVerticesUpdateRequest, \
    SavePictureRequest, SaveCroppedPictureRequest, PictureRequest, CatalogDictRequest, NewCoinRequest, \
    RemoveCoinRequest, DeleteCroppedPicture, UpdateCoinCameraSettingsRequest
from core.qt_threading.messages.catalog_handler.Responses import PictureResponse, \
    CatalogDictResponse


CATALOG_FILE_NAME = "catalog"


class CoinCatalogHandler(QObject):
    def __init__(self):
        super().__init__()
        self.qt_signals: CommonSignals = CommonSignals()
        self.module = Modules.CATALOG_HANDLER
        project_root = 'D:\\Projects\\bachelor_thesis\\OpenCV2-Coin-Recognizer'
        self.catalog_path = os.path.join(project_root, "coin_catalog")
        self.coin_catalog_dict = {}
        self.current_picture: QPixmap | None = None

        self.is_running = False
        self.main_thread = QThread()
        self.process = None

        self.qt_signals.catalog_handler_request.connect(self.handle_request)

    def start_process(self):
        self.moveToThread(self.main_thread)
        self.main_thread.started.connect(self.worker)
        self.main_thread.start()

    def worker(self):
        self.parse_main_catalog()

        # while self.is_running:
        #     QApplication.processEvents()

    def handle_request(self, request: MessageBase):
        request_handlers = {
            CatalogDictRequest: self.handle_coin_catalog_request,
            PictureRequest: self.handle_image_with_vertices_request,
            PictureVerticesUpdateRequest: self.handle_image_vertices_update_request,
            SavePictureRequest: self.handle_save_picture_request,
            SaveCroppedPictureRequest: self.handle_save_cropped_picture_request,
            DeleteCroppedPicture: self.handle_delete_cropped_picture_request,
            # PictureVerticesRequest: self.handle_image_vertices_request,
            NewCoinRequest: self.handle_new_coin_request,
            RemoveCoinRequest: self.handle_remove_coin_request,
            UpdateCoinCameraSettingsRequest: self.handle_update_coin_camera_settings
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def handle_coin_catalog_request(self, request: CatalogDictRequest):
        response = CatalogDictResponse(source=Modules.CATALOG_HANDLER,
                                       destination=request.source,
                                       data=self.coin_catalog_dict)
        self.qt_signals.catalog_handler_response.emit(response)

    def handle_image_with_vertices_request(self, request: PictureRequest):
        _, pic_with_background, pic_no_background = self.get_coin_photo_from_catalog(request.coin, request.picture)
        response = PictureResponse(source=Modules.CATALOG_HANDLER,
                                   destination=request.source,
                                   pic_with_background=pic_with_background,
                                   pic_no_background=pic_no_background)
        self.qt_signals.catalog_handler_response.emit(response)

    def handle_image_vertices_update_request(self, request: PictureVerticesUpdateRequest):
        vertices = request.vertices
        coin = request.coin
        picture_file = request.picture_file
        self.set_coin_photo_vertices(coin=coin, picture_file=picture_file, vertices_coordinates=vertices)
        self.write_catalog()

    def handle_save_picture_request(self, request: SavePictureRequest):
        picture_filename = self.add_coin_picture(image=request.image_with_background, coin=request.coin)
        if request.cropped_image is not None:
            self.add_cropped_coin_picture(image=request.cropped_image, picture_filename=picture_filename, coin=request.coin)

    def handle_save_cropped_picture_request(self, request: SaveCroppedPictureRequest):
        self.add_cropped_coin_picture(image=request.image_without_background,
                                      picture_filename=request.picture_name,
                                      coin=request.coin)

    def handle_delete_cropped_picture_request(self, request: DeleteCroppedPicture):
        self.delete_cropped_coin_picture(picture_filename=request.picture_file,
                                         coin=request.coin)

    # def handle_image_vertices_request(self, request: PictureVerticesRequest):
    #     coin_year: str = request.coin.year
    #     coin_country: str = request.coin.country
    #     coin_name: str = request.coin.name
    #
    #     picture_filename: str = request.picture_filename
    #     # vertices = self.coin_catalog_dict[coin_year][coin_country][coin_name].pictures[picture_filename]["vertices"]
    #
    #     response = PictureVerticesResponse(vertices=vertices,
    #                                        source=Modules.CATALOG_HANDLER,
    #                                        destination=request.source)
    #
    #     self.qt_signals.catalog_handler_request.emit(response)

    def handle_new_coin_request(self, request: NewCoinRequest):
        new_coin = Coin(name=request.coin_name, year=request.coin_year, country=request.coin_country)
        self.add_coin_to_catalog(coin=new_coin)
        self.write_catalog()
        self.handle_coin_catalog_request(CatalogDictRequest(source=self.module, destination=self.module))

    def handle_remove_coin_request(self, request: RemoveCoinRequest):
        self.remove_coin_from_catalog(coin=request.coin)
        self.write_catalog()
        self.handle_coin_catalog_request(CatalogDictRequest(source=self.module, destination=self.module))

    def handle_update_coin_camera_settings(self, request: UpdateCoinCameraSettingsRequest):
        c: Coin = request.coin
        coin = self.coin_catalog_dict[c.year][c.country][c.name]
        coin.contour_detection_params = request.params
        self.write_catalog()

        response = CatalogDictResponse(source=Modules.CATALOG_HANDLER,
                                       destination=request.source,
                                       data=self.coin_catalog_dict)
        self.qt_signals.catalog_handler_response.emit(response)

    def parse_main_catalog(self) -> bool:
        catalog_dict_path = os.path.join(self.catalog_path, CATALOG_FILE_NAME + ".json")
        coin_catalog_dict: dict = {}
        with open(catalog_dict_path, ) as catalog_file:

            file_dict = None
            try:
                file_dict = json.load(catalog_file)
            except json.decoder.JSONDecodeError as ex:
                print(ex)
                return False
            self.global_params = file_dict["params"]
            # print("Params: ", file_dict["params"])

            for year, countries in file_dict["coins"]["years"].items():
                year = year.lower()
                coin_catalog_dict[year] = {}

                for country, coins in countries.items():
                    country = country.lower()
                    coin_catalog_dict[year][country] = {}

                    for coin_name, coin_attributes in coins.items():
                        coin = Coin(name=coin_name, year=year, country=country)

                        try:
                            for param_name, value in coin_attributes["training_params"].items():
                                passed = coin.add_training_param(param_name=param_name, value=value)
                                if not passed:
                                    return False

                            # try reading out CV2 coin contour detection params. Reinit them if none found.
                            contour_detection_params = coin_attributes.get("contour_detection_params")
                            coin.contour_detection_params = ContourDetectionSettings.from_dict(
                                contour_detection_params) if contour_detection_params else ContourDetectionSettings()

                            # Check whether every PNG mentioned in catalog.json exists
                            for picture_file, attributes in coin_attributes["pictures"].items():
                                file_path = os.path.join(self.catalog_path, coin.coin_dir_path(), picture_file)

                                # Do not add link to coin photo if no such file exists.
                                if not os.path.exists(file_path) or not os.path.isfile(file_path):
                                    continue
                                coin.add_picture(picture_file=picture_file.lower())

                                # In case both catalog photo has link to cropped photo outside catalog directory and
                                # cropped photo exists in catalog folder - prefer catalog folder cropped photo.
                                directory_cropped_pic: str = os.path.join(self.catalog_path, "cropped",
                                                                          coin.coin_dir_path(), picture_file)
                                external_cropped_pic: str = attributes.get("cropped_version", None)

                                if os.path.exists(directory_cropped_pic):
                                    coin_attributes["pictures"][picture_file]["cropped_version"] = directory_cropped_pic
                                elif external_cropped_pic and os.path.exists(external_cropped_pic):
                                    coin_attributes["pictures"][picture_file]["cropped_version"] = external_cropped_pic
                                # else:
                                #     coin_attributes["pictures"][picture_file]["cropped_version"] = None

                            # Add PNGs present in coin directory to pictures
                            for filename in os.listdir(os.path.join(self.catalog_path, coin.coin_dir_path())):
                                if filename.lower().endswith('.png') and filename.lower() not in coin.pictures:
                                    coin.add_picture(picture_file=picture_file.lower())

                                    directory_cropped_pic: str = os.path.join(self.catalog_path, "cropped",
                                                                              coin.coin_dir_path(), picture_file)
                                    if os.path.exists(directory_cropped_pic):
                                        coin_attributes["pictures"][picture_file][
                                            "cropped_version"] = directory_cropped_pic

                        except KeyError as ex:
                            print(ex)
                            return False

                    coin_catalog_dict[year][country][coin.name] = coin

        self.coin_catalog_dict = coin_catalog_dict
        return True

    def add_coin_to_catalog(self, coin: Coin):
        try:
            _ = self.coin_catalog_dict[coin.year][coin.country][coin.name]
            # coin already present in catalog
            return
        except KeyError:
            pass
        if coin.year not in self.coin_catalog_dict:
            self.coin_catalog_dict[coin.year] = {}
        if coin.country not in self.coin_catalog_dict[coin.year]:
            self.coin_catalog_dict[coin.year][coin.country] = {}
        self.coin_catalog_dict[coin.year][coin.country][coin.name] = coin

    def remove_coin_from_catalog(self, coin: Coin):
        try:
            _ = self.coin_catalog_dict[coin.year][coin.country][coin.name]
            # coin already present in catalog
        except KeyError:
            return
        for picture in self.coin_catalog_dict[coin.year][coin.country][coin.name].pictures:
            os.remove(os.path.join(self.catalog_path, coin.year, coin.country, coin.name, picture))
        self.coin_catalog_dict[coin.year][coin.country].pop(coin.name)

        if len(self.coin_catalog_dict[coin.year][coin.country].keys()) == 0:
            self.coin_catalog_dict[coin.year].pop(coin.country)

        if len(self.coin_catalog_dict[coin.year].keys()) == 0:
            self.coin_catalog_dict.pop(coin.year)

    def add_coin_picture(self, image: QImage, coin: Coin) -> str:
        pixmap = QPixmap.fromImage(image)
        catalog_coin = self.coin_catalog_dict[coin.year][coin.country][coin.name]
        coin_dir_path: str = os.path.join(self.catalog_path, coin.year, coin.country, coin.name)
        picture_file: str = str(uuid.uuid4()) + ".png"
        absolute_path: str = os.path.join(coin_dir_path, picture_file)

        if not os.path.exists(coin_dir_path):
            os.makedirs(coin_dir_path)

        saved = pixmap.save(absolute_path)
        if saved:
            print(f"image saved to {absolute_path}")
            catalog_coin.add_picture(f"{picture_file}")
            self.write_catalog()
            response = CatalogDictResponse(data=self.coin_catalog_dict)
            self.qt_signals.catalog_handler_response.emit(response)
        return picture_file

    def delete_cropped_coin_picture(self, picture_filename: str, coin: Coin):
        catalog_coin = self.coin_catalog_dict[coin.year][coin.country][coin.name]
        coin_dir_path: str = os.path.join(self.catalog_path, "cropped", catalog_coin.year, catalog_coin.country, catalog_coin.name)
        absolute_path: str = os.path.join(coin_dir_path, picture_filename)

        os.remove(absolute_path)

    def add_cropped_coin_picture(self, image: QImage, picture_filename: str, coin: Coin):
        pixmap = QPixmap.fromImage(image)
        catalog_coin = self.coin_catalog_dict[coin.year][coin.country][coin.name]
        coin_dir_path: str = os.path.join(self.catalog_path, "cropped", catalog_coin.year, catalog_coin.country, catalog_coin.name)
        absolute_path: str = os.path.join(coin_dir_path, picture_filename)

        if not os.path.exists(coin_dir_path):
            os.makedirs(coin_dir_path)

        saved = pixmap.save(absolute_path)
        print(f"add_cropped_coin_picture: {absolute_path}: {saved}")

    def get_coin_dir_picture_files(self, coin: Coin) -> list[str]:
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          coin.year,
                                          coin.country,
                                          coin.name)
        png_file_list: list[str] = [file for file in os.listdir(coin_dir_path) if file.endswith(".png")]
        return [file for file in coin.pictures if file in png_file_list]

    def get_coin_photo_from_catalog(self, coin: Coin, picture: str) -> dict[str, bool | QImage]:
        print(f"[CoinCatalogHandler]: get_coin_photo_from_catalog")
        loaded: bool = False
        pic_with_background: QImage | None = None
        pic_no_background: QImage | None = None
        coin_picture_files = self.get_coin_dir_picture_files(coin)
        # vertices: list[tuple[int, int]] = []

        cropped_dir_path: str = os.path.join(self.catalog_path, "cropped", coin.coin_dir_path())
        coin_dir_path: str = os.path.join(self.catalog_path, coin.coin_dir_path())

        if picture in coin.pictures:
            pic_with_background = QImage(os.path.join(coin_dir_path, picture))
            pic_with_background = pic_with_background.convertToFormat(QImage.Format_RGBA8888)
            if os.path.isfile(os.path.join(cropped_dir_path, picture)):
                pic_no_background = QImage(os.path.join(cropped_dir_path, picture))
                pic_no_background = pic_no_background.convertToFormat(QImage.Format_RGBA8888)

            self.current_picture = QImage(os.path.join(coin_dir_path, picture))
            # vertices = coin.pictures[picture]["vertices"]

            # Check if the image is loaded successfully
            if pic_with_background.isNull():
                print(f"Failed to load image from {picture}")
            else:
                print(f"Image loaded successfully from {coin_picture_files[0]}")
                loaded = True
        else:
            self.current_picture = QImage("data\\debug_img.png",)
        return loaded, pic_with_background, pic_no_background

    def set_coin_photo_vertices(self, coin: Coin, picture_file: str, vertices_coordinates: list[tuple[int, int]]):
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          coin.year,
                                          coin.country,
                                          coin.name)
        png_file_list: list[str] = [file for file in os.listdir(coin_dir_path) if file.endswith(".png")]
        coin.pictures[picture_file]["vertices"] = vertices_coordinates
        self.coin_catalog_dict[coin.year][coin.country][coin.name] = coin
        self.write_catalog()

    def write_catalog(self) -> bool:
        catalog_dict_path = os.path.join(self.catalog_path, CATALOG_FILE_NAME+".json")
        try:
            with open(catalog_dict_path, 'w') as file:
                dictionary: dict = {"params": self.global_params, "coins": {"years": self.coin_catalog_dict}}
                json.dump(dictionary, file, cls=CatalogEncoder, indent=4)
                # TODO: custom serializer
                # json_data = CatalogEncoder.custom_serialize(self.coin_catalog_dict)
                # file.write(json_data)
        except Exception as ex:
            print(ex)
            return False
        return True


class CatalogEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, list):
            return '[' + ', '.join(i for i in obj) + ']'
            # return '[' + ', '.join(CatalogEncoder.custom_serialize(i) for i in obj) + ']\n'
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
