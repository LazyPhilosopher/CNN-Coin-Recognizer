from PySide6.QtCore import QPoint, Slot, QTimer
from PySide6.QtWidgets import QMainWindow, QLabel

from core.catalog.Coin import Coin
from core.catalog.DraggableCrossesOverlay import DraggableCrossesOverlay
from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import RequestBase, Modules
from core.qt_threading.headers.catalog_handler.CatalogDictRequest import CatalogDictRequest
from core.qt_threading.headers.catalog_handler.CatalogDictResponse import CatalogDictResponse
from core.qt_threading.headers.catalog_handler.PictureRequest import PictureRequest
from core.qt_threading.headers.catalog_handler.PictureResponse import PictureResponse
from core.qt_threading.headers.catalog_handler.PictureVerticesUpdateRequest import PictureVerticesUpdateRequest
from core.qt_threading.headers.catalog_handler.SavePictureRequest import SavePictureRequest
from core.qt_threading.headers.video_thread.CameraListRequest import CameraListRequest
from core.qt_threading.headers.video_thread.CameraListResponse import CameraListResponse
from core.qt_threading.headers.video_thread.FrameAvailable import FrameAvailable
from core.ui.ImageFrame import ImageFrame
from core.ui.pyqt6_designer.d_ImageCollector import Ui_ImageCollector


class ImageCollector(QMainWindow, Ui_ImageCollector):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.qt_signals = CommonSignals()

        self.video_frame = ImageFrame(self.video_frame)
        # self.image_label = QLabel(self.video_frame)
        # self.image_label.setGeometry(0, 0, self.video_frame.width(), self.video_frame.height())
        # self.image_label.setScaledContents(True)

        self.catalog: dict | None = None
        self.active_coin: Coin | None = None
        self.current_picture_name: str = ""
        self.image_idx: int = 0

        self.overlay = DraggableCrossesOverlay(self.video_frame)
        # self.overlay.setGeometry(self.video_frame.geometry())
        self.overlay.setGeometry(self.video_frame.rect())

        self.qt_signals.catalog_handler_response.connect(self.receive_request)
        self.qt_signals.video_thread_response.connect(self.receive_request)
        self.next_gallery_photo_button.clicked.connect(self.next_picture_button_callback)
        self.previous_gallery_photo_button.clicked.connect(self.previous_picture_button_callback)
        self.overlay.mouse_released.connect(self.update_edges)
        self.save_photo_button.clicked.connect(self.save_photo)
        self.tabWidget.currentChanged.connect(self.tab_bar_click_routine)

        self.qt_signals.catalog_handler_request.emit(CatalogDictRequest(source=Modules.GALLERY_WINDOW))
        QTimer.singleShot(0, self.request_camera_ids)

    @Slot()
    def receive_request(self, request: RequestBase):
        # print("receive_request")
        current_tab_index = self.tabWidget.currentIndex()

        if isinstance(request, CatalogDictResponse):
            print(f"[ImageGalleryWindow]: {request.catalog}")

            self.catalog = request.catalog
            self.set_year_dropbox_items()

            year = self.coin_catalog_year_dropbox.currentText()
            self.set_country_dropbox_items(year=year)

            country = self.coin_catalog_country_dropbox.currentText()
            self.coin_catalog_name_dropbox.addItems(self.catalog[year][country].keys())

            name = self.coin_catalog_name_dropbox.currentText()
            self.active_coin: Coin = self.catalog[year][country][name]

        elif isinstance(request, CameraListResponse):
            # print(f"[AddNewImageWindow]: {request}")
            self.camera_swich_combo_box.addItems(request.body["cameras"])

        elif isinstance(request, PictureResponse):
            if self.tabWidget.tabText(current_tab_index) == "Gallery":
                picture = request.picture
                vertices = request.vertices
                self.video_frame.set_image(picture)
                width = self.video_frame.width()
                height = self.video_frame.height()
                crosses = [QPoint(x * width, y * height) for (x, y) in vertices]

                # self.overlay.init_image_with_vertices(self.catalog_handler.active_coin, self.current_coin_photo_id)
                self.overlay.crosses = crosses
                self.overlay.show()
                # print(vertices)
                # print(picture)

        if isinstance(request, CatalogDictResponse):
            print(f"[ImageGalleryWindow]: {request.catalog}")

            self.catalog = request.catalog
            self.set_year_dropbox_items()

            year = self.coin_catalog_year_dropbox.currentText()
            self.set_country_dropbox_items(year=year)

            country = self.coin_catalog_country_dropbox.currentText()
            self.coin_catalog_name_dropbox.addItems(self.catalog[year][country].keys())

        elif isinstance(request, FrameAvailable):
            if self.tabWidget.tabText(current_tab_index) == "Camera":
                self.video_frame.set_image(request.frame)

    def tab_bar_click_routine(self):
        current_tab_index = self.tabWidget.currentIndex()
        if self.tabWidget.tabText(current_tab_index) == "Gallery":
            year = self.coin_catalog_year_dropbox.currentText()
            country = self.coin_catalog_country_dropbox.currentText()
            name = self.coin_catalog_name_dropbox.currentText()

            self.active_coin: Coin = self.catalog[year][country][name]
            self.image_idx %= len(self.active_coin.pictures.keys())
            self.current_picture_name = list(self.active_coin.pictures.keys())[self.image_idx]
            self.qt_signals.catalog_handler_request.emit(PictureRequest(coin=self.active_coin,
                                                                        picture=self.current_picture_name,
                                                                        source=Modules.GALLERY_WINDOW,
                                                                        destination=Modules.CATALOG_HANDLER))
        if self.tabWidget.tabText(current_tab_index) == "Camera":
            self.overlay.crosses = []
            self.overlay.show()

    @Slot()
    def request_camera_ids(self):
        print("request_camera_ids")
        self.qt_signals.video_thread_request.emit(CameraListRequest())

    def next_picture_button_callback(self):
        print(f"Next button")
        self.image_idx += 1
        self.request_picture()

    def previous_picture_button_callback(self):
        print(f"Previous button")
        self.image_idx -= 1
        self.request_picture()

    def set_year_dropbox_items(self):
        self.coin_catalog_year_dropbox.blockSignals(True)
        self.coin_catalog_year_dropbox.clear()
        self.coin_catalog_year_dropbox.addItems(list(self.catalog.keys()))
        self.coin_catalog_year_dropbox.blockSignals(False)

    def set_country_dropbox_items(self, year: str):
        self.coin_catalog_country_dropbox.blockSignals(True)
        self.coin_catalog_country_dropbox.clear()
        self.coin_catalog_country_dropbox.addItems(list(self.catalog[year].keys()))
        self.coin_catalog_country_dropbox.blockSignals(False)

    def set_coin_name_dropbox_items(self, year: str, country: str):
        self.coin_catalog_name_dropbox.blockSignals(True)
        self.coin_catalog_name_dropbox.clear()
        self.coin_catalog_name_dropbox.addItems(self.catalog[year][country].keys())
        self.coin_catalog_name_dropbox.blockSignals(False)

    def request_picture(self):
        year = self.coin_catalog_year_dropbox.currentText()
        country = self.coin_catalog_country_dropbox.currentText()
        name = self.coin_catalog_name_dropbox.currentText()

        self.active_coin: Coin = self.catalog[year][country][name]
        self.image_idx %= len(self.active_coin.pictures.keys())
        self.current_picture_name = list(self.active_coin.pictures.keys())[self.image_idx]
        self.qt_signals.catalog_handler_request.emit(PictureRequest(coin=self.active_coin,
                                                                    picture=self.current_picture_name,
                                                                    source=Modules.GALLERY_WINDOW,
                                                                    destination=Modules.CATALOG_HANDLER))

    def save_photo(self):
        request = SavePictureRequest(picture=self.video_frame.image_label.pixmap(),
                                     coin=self.active_coin)
        self.qt_signals.catalog_handler_request.emit(request)

    def update_edges(self, crosses: list[QPoint]):
        width = self.video_frame.width()
        height = self.video_frame.height()
        vertices = [(point.x() / width, point.y() / height) for point in crosses]
        request = PictureVerticesUpdateRequest(source=Modules.DRAGGABLE_CROSS_OVERLAY,
                                               destination=Modules.CATALOG_HANDLER,
                                               coin=self.active_coin,
                                               vertices=vertices,
                                               picture_file=self.current_picture_name)
        self.qt_signals.catalog_handler_request.emit(request)

