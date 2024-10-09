import json
import os
import uuid

from PySide6.QtCore import QObject, QThread
from PySide6.QtGui import QPixmap, QImage

from core.modules.catalog.Coin import Coin
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import MessageBase, Modules
from core.qt_threading.messages.catalog_handler.Requests import PictureVerticesRequest, PictureVerticesUpdateRequest, \
    SavePictureRequest, PictureRequest, CatalogDictRequest, NewCoinRequest, RemoveCoinRequest
from core.qt_threading.messages.catalog_handler.Responses import PictureVerticesResponse, PictureResponse, \
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
                        coin = Coin(coin_name)
                        coin.year = year
                        coin.country = country

                        try:
                            for param_name, value in coin_attributes["training_params"].items():
                                passed = coin.add_training_param(param_name=param_name, value=value)
                                if not passed:
                                    return False
                            for param_name, value in coin_attributes["coin_params"].items():
                                passed = coin.add_coin_param(param_name=param_name, value=value)
                                if not passed:
                                    return False
                            for picture_file, attributes in coin_attributes["pictures"].items():
                                file_path = os.path.join(self.catalog_path, coin.year, coin.country, coin.name, picture_file)
                                if not os.path.exists(file_path) or not os.path.isfile(file_path):
                                    continue
                                coin.add_picture(picture_file=picture_file)
                                vertices: list[list[float, float]] = attributes["vertices"]
                                passed = coin.add_vertices_to_picture(picture_file=picture_file, vertices=vertices)
                                if not passed:
                                    return False
                        except KeyError as ex:
                            print(ex)
                            return False

                    coin_catalog_dict[year][country][coin.name] = coin

        self.coin_catalog_dict = coin_catalog_dict
        return True

    def handle_request(self, request: MessageBase):
        request_handlers = {
            CatalogDictRequest: self.handle_coin_catalog_request,
            PictureRequest: self.handle_image_with_vertices_request,
            PictureVerticesUpdateRequest: self.handle_image_vertices_update_request,
            SavePictureRequest: self.handle_save_picture_request,
            PictureVerticesRequest: self.handle_image_vertices_request,
            NewCoinRequest: self.handle_new_coin_request,
            RemoveCoinRequest: self.handle_remove_coin_request
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
        picture, vertices = self.get_coin_photo_from_catalog(request.coin, request.picture)
        response = PictureResponse(source=Modules.CATALOG_HANDLER,
                                   destination=request.source,
                                   picture=picture,
                                   vertices=vertices)
        self.qt_signals.catalog_handler_response.emit(response)

    def handle_image_vertices_update_request(self, request: PictureVerticesUpdateRequest):
        vertices = request.vertices
        coin = request.coin
        picture_file = request.picture_file
        self.set_coin_photo_vertices(coin=coin, picture_file=picture_file, vertices_coordinates=vertices)
        self.write_catalog()

    def handle_save_picture_request(self, request: SavePictureRequest):
        self.add_coin_picture(image=request.image,
                              coin=request.coin)

    def handle_image_vertices_request(self, request: PictureVerticesRequest):
        coin_year: str = request.coin.year
        coin_country: str = request.coin.country
        coin_name: str = request.coin.name

        picture_filename: str = request.picture_filename
        vertices = self.coin_catalog_dict[coin_year][coin_country][coin_name].pictures[picture_filename]["vertices"]

        response = PictureVerticesResponse(vertices=vertices,
                                           source=Modules.CATALOG_HANDLER,
                                           destination=request.source)

        self.qt_signals.catalog_handler_request.emit(response)

    def handle_new_coin_request(self, request: NewCoinRequest):
        new_coin = Coin(name=request.coin_name, year=request.coin_year, country=request.coin_country)
        self.add_coin_to_catalog(coin=new_coin)
        self.write_catalog()
        self.handle_coin_catalog_request(CatalogDictRequest(source=self.module, destination=self.module))

    def handle_remove_coin_request(self, request: RemoveCoinRequest):
        self.remove_coin_from_catalog(coin=request.coin)
        self.write_catalog()
        self.handle_coin_catalog_request(CatalogDictRequest(source=self.module, destination=self.module))

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

    def add_coin_picture(self, image: QImage, coin: Coin):
        pixmap = image.pixmap()
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

    def get_coin_dir_picture_files(self, coin: Coin) -> list[str]:
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          coin.year,
                                          coin.country,
                                          coin.name)
        png_file_list: list[str] = [file for file in os.listdir(coin_dir_path) if file.endswith(".png")]
        return [file for file in coin.pictures if file in png_file_list]

    def get_coin_photo_from_catalog(self, coin: Coin, picture: str) -> tuple[QPixmap, list[tuple[float, float]]]:
        print(f"[CoinCatalogHandler]: get_coin_photo_from_catalog")
        coin_picture_files = self.get_coin_dir_picture_files(coin)
        vertices: list[tuple[int, int]] = []
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          coin.year,
                                          coin.country,
                                          coin.name)
        if picture in coin.pictures:
            self.current_picture = QPixmap(os.path.join(coin_dir_path, picture))
            vertices = coin.pictures[picture]["vertices"]
            self.current_picture = self.current_picture

            # Check if the image is loaded successfully
            if self.current_picture.isNull():
                print(f"Failed to load image from {picture}")
            else:
                print(f"Image loaded successfully from {coin_picture_files[0]}")
        else:
            self.current_picture = QPixmap("data\\debug_img.png")
        return self.current_picture, vertices

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
