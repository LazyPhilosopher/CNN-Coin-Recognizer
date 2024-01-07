import numpy
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QLabel

from core.catalog.Coin import Coin
from core.threading.signals import ThreadingSignals
from core.ui.NewCoinWidget import NewCoinWidget
from core.ui.pyqt6_designer.d_ImageCollector import Ui_w_ImageCollector


class ImageCollector(QWidget, Ui_w_ImageCollector):
    def __init__(self, signals: ThreadingSignals):
        super().__init__()
        self.signals: ThreadingSignals = signals
        self.setupUi(self)

        self.image_label = QLabel(self.video_frame)
        self.image_label.setGeometry(0, 0, self.video_frame.width(), self.video_frame.height())
        self.image_label.setScaledContents(True)  # Ensure image scales with QLabel
        self.image_label.setPixmap(QPixmap("data\\debug_img.png"))
        self.new_coin_window = None

        self._coin_list: list[Coin] = None

        # self.signals.s_catalog_changed.connect(self.refresh_coins_combo_box)
        self.new_coin_button.pressed.connect(self.new_coin_button_press)
        self.save_photo_button.pressed.connect(self.save_coin_photo_button_routine)
        self.camera_swich_combo_box.currentIndexChanged.connect(self.context_box_pressed_routine)
        self.active_coin_combo_box.currentIndexChanged.connect(self.active_coin_combo_box_changed_routine)

        self.next_gallery_photo_button.pressed.connect(self.next_photo_routine)
        self.previous_gallery_photo_button.pressed.connect(self.previous_photo_routine)

        self.signals.s_append_info_text.connect(self.append_info_text)
        self.color_correction_button.pressed.connect(self.color_correction_routine)
        self.tabWidget.currentChanged.connect(self.tab_switch_routine)

        self.vertices_reset_button.pressed.connect(self.mark_reset_button_routine)

    def refresh_camera_combo_box(self, camera_list: list[str]):
        self.camera_swich_combo_box.clear()
        self.camera_swich_combo_box.addItems(camera_list)

    def context_box_pressed_routine(self, event):
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

    def save_coin_photo_button_routine(self):
        self.signals.s_save_picture_button_pressed.emit()

    def append_info_text(self, message: str):
        self.plainTextEdit.appendPlainText(message)

    def color_correction_routine(self):
        self.signals.s_append_info_text.emit("Color correction button clicked!")

    def refresh_coins_combo_box(self, coin_list: list[Coin]):
        self._coin_list = coin_list
        coin_name_list: list[str] = [coin.name for coin in self._coin_list]
        self.active_coin_combo_box.clear()
        self.active_coin_combo_box.addItems(coin_name_list)

    def set_active_coin(self, active_coin: Coin):
        try:
            idx: int = self._coin_list.index(active_coin)
            self.active_coin_combo_box.setCurrentIndex(idx)
        except Exception as ex:
            print(f"ERROR: such coin not found in context box! {ex}")

    def active_coin_combo_box_changed_routine(self):
        idx: int = self.active_coin_combo_box.currentIndex()
        new_active_coin: Coin = self._coin_list[idx]
        self.signals.s_active_coin_changed.emit(new_active_coin)

    def tab_switch_routine(self):
        self.signals.s_active_tab_changed.emit()

    def next_photo_routine(self):
        self.signals.s_coin_photo_id_changed.emit(1)

    def previous_photo_routine(self):
        self.signals.s_coin_photo_id_changed.emit(-1)

    def mark_reset_button_routine(self):
        self.signals.s_reset_vertices.emit()
