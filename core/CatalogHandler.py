import json
import json
import os
import uuid

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen

from core.catalog.Coin import Coin
from core.threading.signals import ThreadingSignals

CATALOG_FILE_NAME = "catalog"


class CoinCatalogHandler:
    catalog_dict_path: str
    coin_catalog_dict: dict = {}
    global_params: dict = {}
    active_coin: Coin = None

    def __init__(self, signals):
        self.signals: ThreadingSignals = signals
        # self.main_catalog = {}
        # self.active_catalog = {}

        # project_root = 'C:\\Users\\Call_me_Utka\\Desktop\\OpenCV2-Coin-Recognizer'
        project_root = 'C:\\Users\\Call_me_Utka\\Desktop\\OpenCV2-Coin-Recognizer'
        self.catalog_path = os.path.join(project_root, "coin_catalog")
        self.signals.new_coin_created.connect(self.add_new_coin_to_catalog)
        # self.current_photo_id: int = 0

        if not self.parse_main_catalog():
            self.init_catalog()
        self.check_catalog_dir()
        self.write_catalog()

    def parse_main_catalog(self) -> bool:
        catalog_dict_path = os.path.join(self.catalog_path, CATALOG_FILE_NAME+".json")
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
                                coin.add_picture(picture_file=picture_file)
                                vertices: list[list[int, int]] = attributes["vertices"]
                                passed = coin.add_vertices_to_picture(picture_file=picture_file, vertices=vertices)
                                if not passed:
                                    return False
                        except KeyError as ex:
                            print(ex)
                            return False

                    coin_catalog_dict[year][country][coin.name] = coin

        self.coin_catalog_dict = coin_catalog_dict
        return True

    # def parse_coin_config(self) -> bool:
    #     return True
    def init_catalog(self) -> bool:
        coin_catalog_dict: dict = {
            "params": {},
            "coins": {
                "years": {}
            }
        }
        self.coin_catalog_dict = coin_catalog_dict
        return True

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

    def check_catalog_dir(self) -> bool:
        for year, countries in self.coin_catalog_dict.items():
            for country, coins in countries.items():
                for coin_name, coin_data in coins.items():
                    coin_dir_path: str = os.path.join(self.catalog_path, year, country, coin_name)
                    os.makedirs(coin_dir_path, exist_ok=True)
                    coin_pictures_not_found: list[str] = []
                    for coin_picture in coin_data.pictures:
                        picture_file_path: str = os.path.join(self.catalog_path, year, country, coin_name, coin_picture)
                        if not(os.path.exists(picture_file_path) and os.path.isfile(picture_file_path)):
                            print(f"ERROR: {picture_file_path} not found!")
                            coin_pictures_not_found.append(coin_picture)
                    [coin_data.pictures.pop(picture) for picture in coin_pictures_not_found]
        return True

    def add_new_coin_to_catalog(self, coin_info_dictionary):
        try:
            name = coin_info_dictionary["name"]
            year = coin_info_dictionary["year"]
            country = coin_info_dictionary['country']
            weight = coin_info_dictionary['weight']
            content = coin_info_dictionary['content']

            coin = Coin(name)
            coin.year = year.lower()
            coin.country = country.lower()
            coin.coin_params["weight"] = weight
            coin.coin_params["content"] = content

            if coin.year not in self.coin_catalog_dict:
                self.coin_catalog_dict[coin.year] = {}
            if coin.country not in self.coin_catalog_dict[coin.year]:
                self.coin_catalog_dict[coin.year][coin.country] = {}

            self.coin_catalog_dict[year][country][coin.name] = coin
            self.write_catalog()
            self.signals.info_signal.emit(True, {"receiver_id": coin_info_dictionary["sender_id"]})
            self.signals.s_catalog_changed.emit()
        except Exception as ex:
            self.signals.info_signal.emit(False, {"receiver_id": coin_info_dictionary["sender_id"], "message": ex})

    def get_list_of_coins(self) -> list[Coin]:
        out = []
        for year, countries in self.coin_catalog_dict.items():
            for country, coins in countries.items():
                for coin_name, data in coins.items():
                    out.append(data)

        return out

    def set_active_coin(self, coin: Coin):
        self.active_coin: Coin = coin

    def add_coin_image(self, image: QPixmap):
        year: str = self.active_coin.year
        country: str = self.active_coin.country
        coin_dir_path: str = os.path.join(self.catalog_path, year, country, self.active_coin.name)
        picture_file: str = str(uuid.uuid4()) + ".png"
        absolute_path: str = os.path.join(coin_dir_path, picture_file)
        saved = image.save(absolute_path, "PNG")
        if saved:
            print(f"image saved to {absolute_path}")
            self.active_coin.add_picture(f"{picture_file}")
            self.write_catalog()

    def get_nth_coin_photo_from_catalog(self, active_coin_photo_id: int) -> QPixmap:
        coin_dir_path: str = os.path.join(self.catalog_path,
                                          self.active_coin.year,
                                          self.active_coin.country,
                                          self.active_coin.name)
        photo_file_list: list[str] = [file for file in os.listdir(coin_dir_path) if file.endswith(".png")]
        active_coin_photo_id = active_coin_photo_id % len(photo_file_list)
        photo_file: str = photo_file_list[active_coin_photo_id]
        photo = QPixmap(os.path.join(coin_dir_path, photo_file))

        # Check if the image is loaded successfully
        if photo.isNull():
            print(f"Failed to load image from {photo_file}")
        else:
            print(f"Image loaded successfully from {photo_file}")

        # Points as percent of the pixmap's dimensions [(x1, y1), (x2, y2), ...]
        points_percent = [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)]

        # Convert percentage coordinates to pixel coordinates
        points_pixel = [(x * photo.width(), y * photo.height()) for x, y in points_percent]

        # Create a QPainter object and start the painting process
        painter = QPainter(photo)
        painter.setPen(QPen(Qt.black, 3))  # Set pen color and thickness

        # Draw lines between the points
        for i in range(len(points_pixel) - 1):
            painter.drawLine(points_pixel[i][0], points_pixel[i][1],
                             points_pixel[i + 1][0], points_pixel[i + 1][1])

        # Optionally, close the shape by drawing a line between the last and first point
        painter.drawLine(points_pixel[-1][0], points_pixel[-1][1],
                         points_pixel[0][0], points_pixel[0][1])

        # End the painting process
        painter.end()
        return photo


    # def post_init(self):
    #     if not os.path.exists(self.catalog_dict_path):
    #         self.main_catalog = {'coins': {}}
    #         self.save_catalog()
    #     self.load_catalog()

    # def add_new_coin_to_catalog(self, coin_dict):
    #     coin_name = coin_dict['name']
    #     coin_year = coin_dict['year']
    #     coin_country = coin_dict['country']
    #     coin_weight = coin_dict['weight']
    #     coin_shape_dict = {}
    #
    #     coin_folder_name = f"{coin_name}___{coin_country}___{coin_year}"
    #     folder_path = os.path.join(self.catalog_path, coin_folder_name)
    #
    #     if coin_folder_name not in self.main_catalog:
    #         self.main_catalog['coins'][coin_folder_name] = {}
    #         self.main_catalog['coins'][coin_folder_name]['name'] = coin_name
    #         self.main_catalog['coins'][coin_folder_name]['year'] = coin_year
    #         self.main_catalog['coins'][coin_folder_name]['country'] = coin_country
    #         self.main_catalog['coins'][coin_folder_name]['weight'] = coin_weight
    #         self.main_catalog['coins'][coin_folder_name]['directory'] = folder_path
    #
    #     if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
    #         os.mkdir(folder_path)
    #
    #     coin_data_dict = os.path.join(folder_path, 'data.json')
    #     try:
    #         with open(coin_data_dict, 'r') as json_file:
    #             coin_folder_catalog = json.load(json_file)
    #
    #     except:
    #         coin_folder_catalog = {'name': coin_name,
    #                                'year': coin_year,
    #                                'country': coin_country,
    #                                'weight': coin_weight,
    #                                'image_files': {}
    #                                }
    #         with open(coin_data_dict, 'w') as json_file:
    #             json.dump(coin_folder_catalog, json_file)
    #
    #     self.save_catalog()

    # def add_coin_image(self, pixmap):
    #     if self.active_coin_name is not None:
    #         coin_dir_path = self.main_catalog['coins'][self.active_coin_name]['directory']
    #
    #         if os.path.exists(coin_dir_path) and os.path.isdir(coin_dir_path):
    #             pic_name = str(uuid.uuid4())
    #             with open(os.path.join(coin_dir_path, 'data.json'), 'r') as json_file:
    #                 self.active_catalog = json.load(json_file)
    #             self.active_catalog['image_files'][pic_name] = {'edges': []}
    #             pixmap.save(os.path.join(coin_dir_path, pic_name + '.PNG'))
    #             with open(os.path.join(coin_dir_path, 'data.json'), 'w') as json_file:
    #                 json.dump(self.active_catalog, json_file)

    # def save_catalog(self):
    #     with open(self.catalog_dict_path, 'w') as json_file:
    #         json.dump(self.main_catalog, json_file)
    #         # CatalogEncoder.custom_serialize(self.main_catalog, json_file)
    #     self.signals.set_coins_combo_box.emit(list(self.main_catalog['coins'].keys()))

    # def load_catalog(self):
    #     with open(self.catalog_dict_path, 'r') as json_file:
    #         self.main_catalog = json.load(json_file)
    #     self.signals.set_coins_combo_box.emit(list(self.main_catalog['coins'].keys()))

    # def set_active_catalog(self, coin_name):
    #     self.active_coin_name = coin_name
    #     if self.active_coin_name is None:
    #         return
    #     active_catalog_dir = self.main_catalog['coins'][self.active_coin_name]['directory']
    #     with open(active_catalog_dir, 'r') as json_file:
    #         self.active_catalog = json.load(json_file)


class CatalogEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, list):
            return '[' + ', '.join(i for i in obj) + ']'
            # return '[' + ', '.join(CatalogEncoder.custom_serialize(i) for i in obj) + ']\n'
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

    # @staticmethod
    # def custom_serialize(data):
    #     if isinstance(data, dict):
    #         items = ",\n".join(f'    "{k}": {CatalogEncoder.custom_serialize(v)}' for k, v in data.items())
    #         return "{\n" + items + "\n}"
    #     elif isinstance(data, list):
    #         return '[' + ', '.join(CatalogEncoder.custom_serialize(i) for i in data) + ']'
    #     elif hasattr(data, 'to_dict'):
    #         return data.to_dict()
    #     else:
    #         return json.dumps(data)
