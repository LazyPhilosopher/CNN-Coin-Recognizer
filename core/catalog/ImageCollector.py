import sys
import typing
import numpy
import os
import json
import uuid
from PySide6.QtCore import QRectF, Qt, QCoreApplication
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QFrame, QWidget, QLabel, \
    QComboBox, QPushButton
from PySide6.QtGui import QPixmap, QScreen, QImage

from core.catalog.Coin import Coin
from core.ui.pyqt6_designer.d_ImageCollector import Ui_w_ImageCollector
from core.ui.NewCoinWidget import NewCoinWidget
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
        # self.catalog_dict_path = os.path.join(self.catalog_path, "catalog_reserve.json")

        if not self.parse_main_catalog():
            self.init_catalog()
            # self.write_catalog()
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
                            print(f"ERROR: {coin_picture} not found!")
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
        # coin = self.active_coin

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


class ImageCollector(QWidget, Ui_w_ImageCollector):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.setupUi(self)

        self.image_label = QLabel(self.video_frame)
        self.image_label.setGeometry(0, 0, self.video_frame.width(), self.video_frame.height())
        self.image_label.setScaledContents(True)  # Ensure image scales with QLabel
        self.image_label.setPixmap(QPixmap("data\\debug_img.png"))
        self.mark_button.setEnabled(False)
        self.new_coin_window = None
        self.catalog_handler = CoinCatalogHandler(self.signals)
        self.refresh_coins_combo_box()

        self.signals.s_catalog_changed.connect(self.refresh_coins_combo_box)
        self.new_coin_button.pressed.connect(self.new_coin_button_press)
        self.save_photo_button.pressed.connect(self.save_coin_photo)
        self.camera_swich_combo_box.currentIndexChanged.connect(self.context_box_press)
        self.active_coin_combo_box.currentIndexChanged.connect(self.set_coins_combo_box)
        self.signals.s_append_info_text.connect(self.append_info_text)

    def set_camera_combo_box(self, camera_list):
        self.camera_swich_combo_box.addItems(camera_list)

    def refresh_coins_combo_box(self):
        self.active_coin_combo_box.clear()
        coin_list: list[Coin] = self.catalog_handler.get_list_of_coins()
        coin_name_list: list[str] = [coin.name for coin in coin_list]
        self.active_coin_combo_box.addItems(coin_name_list)
        try:
            current_idx: int = self.active_coin_combo_box.currentIndex()
            self.catalog_handler.set_active_coin(coin_list[current_idx])
            self.active_coin_combo_box.setCurrentIndex(current_idx)
            self.plainTextEdit.appendPlainText(f"Current coin: {self.catalog_handler.active_coin.name}")

        except Exception as ex:
            self.catalog_handler.active_coin_name = None

    def context_box_press(self, event):
        # print(f"CONTEXT BOX PRESSED {event}")
        self.signals.camera_reinit_signal.emit(event)

    def set_image(self, image_frame):
        match image_frame:
            case QPixmap():
                pic = image_frame
            case numpy.ndarray():
                height, width, channel = image_frame.shape
                bytes_per_line = 3 * width
                qimage = QImage(image_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Convert the QImage to a QPixmap
                pic = QPixmap.fromImage(qimage)
            case _:
                pic = QPixmap("data\\debug_img.png")

        # Set the aspectRatioMode property to KeepAspectRatio
        self.image_label.setPixmap(pic)
        # self.image_label.setScaledContents(True)

    def new_coin_button_press(self):
        self.new_coin_window = NewCoinWidget(signals=self.signals)
        self.new_coin_window.setWindowTitle("Create new Coin")
        self.new_coin_window.show()

    def save_coin_photo(self):
        if self.catalog_handler.active_coin is not None:
            self.catalog_handler.add_coin_image(self.image_label.pixmap())

    def set_coins_combo_box(self):
        idx: int = self.active_coin_combo_box.currentIndex()
        active_coin: Coin = self.catalog_handler.get_list_of_coins()[idx]
        self.catalog_handler.set_active_coin(active_coin)
        self.append_info_text(f"Active coin: {active_coin.name}")

    def append_info_text(self, message: str):
        self.plainTextEdit.appendPlainText(message)
    #     if self.folder_name.text() == "":
    #         return
    #
    #     project_root = 'C:\\Users\\Call_me_Utka\\Desktop\\OpenCV2-Coin-Recognizer'
    #
    #     # Relative path to the folder you want to check
    #     folder_path = os.path.join(project_root, "coin_catalog", self.folder_name.text())
    #
    #     # Check if the folder exists
    #     if os.path.exists(folder_path) and os.path.isdir(folder_path):
    #         print(f"New Photo button pressed. {event}")
    #     else:
    #         os.mkdir(folder_path)
    #
    #     dir_list = os.listdir(folder_path)
    #     if self.next_coin_name.text() == "":
    #         self.next_coin_name.setText(f"{self.folder_name.text()}_{len(dir_list)}.png")
    #     self.image_label.pixmap().save(os.path.join(folder_path, self.next_coin_name.text()), 'PNG')
    #
    #     self.next_coin_name.setText(f"{self.folder_name.text()}_{len(dir_list)+1}.png")


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
