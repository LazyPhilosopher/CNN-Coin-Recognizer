import json
import os

from PySide6.QtCore import QObject, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from core.catalog.Coin import Coin
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import RequestBase, Modules
from core.qt_threading.headers.catalog_handler.CatalogDictRequest import CatalogDictRequest
from core.qt_threading.headers.catalog_handler.CatalogDictResponse import CatalogDictResponse
from core.qt_threading.headers.catalog_handler.PictureRequest import PictureRequest
from core.qt_threading.headers.catalog_handler.PictureResponse import PictureResponse
from core.qt_threading.headers.catalog_handler.PictureVerticesUpdateRequest import PictureVerticesUpdateRequest

CATALOG_FILE_NAME = "catalog"


class CoinCatalogHandler(QObject):
    def __init__(self):
        super().__init__()
        self.qt_signals: CommonSignals = CommonSignals()
        project_root = 'D:\\Projects\\bachelor_thesis\\OpenCV2-Coin-Recognizer'
        self.catalog_path = os.path.join(project_root, "coin_catalog")
        self.coin_catalog_dict = {}

        self.is_running = False
        self.main_thread = QThread()
        self.process = None

        self.qt_signals.catalog_handler_request.connect(self.receive_request)

    def start_process(self):
        self.moveToThread(self.main_thread)
        self.main_thread.started.connect(self.worker)
        self.main_thread.start()

    def worker(self):
        self.parse_main_catalog()

        while self.is_running:
            QApplication.processEvents()

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

    def get_coin_dir_picture_files(self, coin: Coin) -> list[str]:
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          coin.year,
                                          coin.country,
                                          coin.name)
        png_file_list: list[str] = [file for file in os.listdir(coin_dir_path) if file.endswith(".png")]
        return [file for file in coin.pictures if file in png_file_list]

    def get_coin_photo_from_catalog(self, coin: Coin, picture: str) -> tuple[QPixmap, list[tuple[float, float]]]:
        # coin: Coin = self.coin_catalog_dict[year][country][name]
        coin_picture_files = self.get_coin_dir_picture_files(coin)
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          coin.year,
                                          coin.country,
                                          coin.name)
        if picture in coin.pictures:
            # photo_file: str = photo_file_list[active_coin_photo_id]
            photo = QPixmap(os.path.join(coin_dir_path, picture))
            vertices: list[tuple[int, int]] = coin.pictures[picture]["vertices"]
            # Check if the image is loaded successfully
            if photo.isNull():
                print(f"Failed to load image from {picture}")
            else:
                print(f"Image loaded successfully from {coin_picture_files[0]}")
        else:
            photo = QPixmap("data\\debug_img.png")
        return photo, vertices

    def set_coin_photo_vertices(self, coin: Coin, picture_file: str, vertices_coordinates: list[tuple[int, int]]):
        # coin: Coin = coin
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          coin.year,
                                          coin.country,
                                          coin.name)
        png_file_list: list[str] = [file for file in os.listdir(coin_dir_path) if file.endswith(".png")]
        coin.pictures[picture_file]["vertices"] = vertices_coordinates
        self.coin_catalog_dict[coin.year][coin.country][coin.name] = coin
        self.write_catalog()
        # photo_id = photo_id % len(coin_picture_files)
        # photo_file: str = photo_file_list[active_coin_photo_id]
        # photo = QPixmap(os.path.join(coin_dir_path, coin_picture_files[photo_id]))
        # coin.pictures[coin_picture_files[photo_id]][
        #     "vertices"] = vertices_coordinates

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

    def receive_request(self, request: RequestBase):
        print(f"[CoinCatalogHandler]: got request: {request}")
        if isinstance(request, CatalogDictRequest):
            response = CatalogDictResponse(source=Modules.CATALOG_HANDLER, destination=request.source, data=self.coin_catalog_dict)
            self.qt_signals.catalog_handler_response.emit(response)
        elif isinstance(request, PictureRequest):
            picture, vertices = self.get_coin_photo_from_catalog(request.coin, request.picture)
            response = PictureResponse(source=Modules.CATALOG_HANDLER,
                                       destination=request.source,
                                       picture=picture,
                                       vertices=vertices)
            self.qt_signals.catalog_handler_response.emit(response)
        elif isinstance(request, PictureVerticesUpdateRequest):
            vertices = request.vertices
            coin = request.coin
            picture_file = request.picture_file
            self.set_coin_photo_vertices(coin=coin, picture_file=picture_file, vertices_coordinates=vertices)
            self.write_catalog()


class CatalogEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, list):
            return '[' + ', '.join(i for i in obj) + ']'
            # return '[' + ', '.join(CatalogEncoder.custom_serialize(i) for i in obj) + ']\n'
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
