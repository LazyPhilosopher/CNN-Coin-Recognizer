from PySide6.QtCore import QPoint, Slot, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMainWindow

from core.designer.pyqt6_designer.NewCoinDialog import NewCoinDialog
from core.modules.catalog.Coin import Coin
from core.modules.catalog.DraggableCrossesOverlay import DraggableCrossesOverlay
from core.qt_threading.common_signals import CommonSignals, blocking_response_message_await
from core.qt_threading.messages.MessageBase import MessageBase, Modules
from core.qt_threading.messages.catalog_handler.Requests import CatalogDictRequest, \
    PictureRequest, SavePictureRequest, PictureVerticesUpdateRequest, NewCoinRequest, RemoveCoinRequest, \
    SaveCroppedPictureRequest, SaveVerticeCropPictureRequest, DeleteCroppedPicture
from core.qt_threading.messages.catalog_handler.Responses import CatalogDictResponse, \
    PictureResponse
from core.qt_threading.messages.processing_module.RemoveBackgroundDictionary import RemoveBackgroundDictionary
from core.qt_threading.messages.processing_module.Requests import GrayscalePictureRequest, RemoveBackgroundRequest, \
    RemoveBackgroundVerticesRequest
from core.qt_threading.messages.processing_module.Responses import ProcessedImageResponse
from core.qt_threading.messages.video_thread.Requests import FrameAvailable, CameraListMessage
from core.qt_threading.messages.video_thread.Responses import CameraListResponse
from core.designer.ImageFrame import ImageFrame
from core.designer.pyqt6_designer.d_ImageCollector import Ui_ImageCollector


class ImageCollector(QMainWindow, Ui_ImageCollector):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.qt_signals = CommonSignals()
        self.module: Modules = Modules.IMAGE_COLLECTOR_WINDOW

        self.video_frame = ImageFrame(self.video_frame)
        # self.image_label = QLabel(self.video_frame)
        # self.image_label.setGeometry(0, 0, self.video_frame.width(), self.video_frame.height())
        # self.image_label.setScaledContents(True)

        self.catalog: dict | None = None
        self.active_coin: Coin | None = None

        self.uncropped_image: QImage | None = None
        self.cropped_picture: QImage | None = None

        self.current_picture_name: str = ""
        self.image_idx: int = 0

        self.overlay = DraggableCrossesOverlay(self.video_frame)
        self.overlay.setGeometry(self.video_frame.rect())

        self.qt_signals.catalog_handler_response.connect(self.handle_request)
        self.qt_signals.video_thread_request.connect(self.handle_request)
        self.qt_signals.frame_available.connect(self.handle_request)
        self.next_gallery_photo_button.clicked.connect(self.next_picture_button_callback)
        self.previous_gallery_photo_button.clicked.connect(self.previous_picture_button_callback)
        self.manual_contour_reset_button.clicked.connect(self.reset_coin_vertices)
        self.overlay.mouse_released.connect(self.update_edges)
        self.save_photo_button.clicked.connect(self.save_photo)
        self.new_coin_button.clicked.connect(self.new_coin_routine)
        self.remove_coin_button.clicked.connect(self.remove_coin_routine)
        self.tabWidget.currentChanged.connect(self.tab_bar_click_routine)
        self.coin_catalog_year_dropbox.currentIndexChanged.connect(self.year_dropbox_update_callback)
        self.coin_catalog_country_dropbox.currentIndexChanged.connect(self.country_dropbox_update_callback)
        self.redraw_coin_contour_checkbox.stateChanged.connect(self.handle_redraw_coin_contour_checkbox)
        self.automatic_contour_detextion_checkbox.stateChanged.connect(self.handle_automatic_contour_detection_checkbox)

        self.gallery_dilate_1_slider.valueChanged.connect(self.request_background_free_image)
        self.gallery_dilate_2_slider.valueChanged.connect(self.request_background_free_image)
        self.gallery_erode_1_slider.valueChanged.connect(self.request_background_free_image)
        self.gallery_erode_2_slider.valueChanged.connect(self.request_background_free_image)
        self.gallery_blur_kernel_slider.valueChanged.connect(self.request_background_free_image)
        self.gallery_dilate_iter_slider.valueChanged.connect(self.request_background_free_image)
        self.gallery_erode_iter_slider.valueChanged.connect(self.request_background_free_image)
        self.save_cropped_image_button.clicked.connect(self.save_image_without_background)

        self.qt_signals.catalog_handler_request.emit(CatalogDictRequest(source=Modules.GALLERY_WINDOW))
        QTimer.singleShot(0, self.request_camera_ids)

    def handle_request(self, request: MessageBase):
        request_handlers = {
            CatalogDictResponse: self.handle_catalog_dict_response,
            CameraListResponse: self.handle_camera_list_response,
            PictureResponse: self.handle_picture_response,
            FrameAvailable: self.handle_frame_available_request
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def handle_catalog_dict_response(self, request: CatalogDictResponse):
        print(f"[ImageGalleryWindow]: {request.catalog}")

        self.catalog = request.catalog
        self.reset_dropboxes()
        try:
            # self.update_active_coin()
            print("self.pick_coin_from_dropboxes(self.active_coin)")
            self.pick_coin_from_dropboxes(self.active_coin)
        except (AttributeError, KeyError):
            pass
        self.update_active_coin()
        print("self.update_active_coin()")

    def handle_camera_list_response(self, request: CameraListResponse):
        self.camera_swich_combo_box.addItems(request.cameras)

    def handle_picture_response(self, request: PictureResponse):
        current_tab_index = self.tabWidget.currentIndex()
        if self.tabWidget.tabText(current_tab_index) == "Gallery":
            pic_with_background: QImage = request.pic_with_background
            pic_no_background: QImage = request.pic_no_background

            self.uncropped_image = pic_with_background
            self.cropped_picture = pic_no_background

            if pic_no_background is not None:
                self.video_frame.set_front_image(pic_no_background)
                self.redraw_coin_contour_checkbox.setChecked(False)
                self.redraw_coin_contour_checkbox.setEnabled(True)
            else:
                self.video_frame.set_front_image(pic_with_background)
                self.redraw_coin_contour_checkbox.setChecked(True)
                self.redraw_coin_contour_checkbox.setEnabled(False)


            # width = self.video_frame.width()
            # height = self.video_frame.height()
            # crosses = [QPoint(x * width, y * height) for (x, y) in vertices]

            # self.overlay.crosses = crosses
            # self.overlay.show()

    def handle_frame_available_request(self, request: FrameAvailable):
        current_tab_index = self.tabWidget.currentIndex()
        if self.tabWidget.tabText(current_tab_index) == "Camera":
            if self.auto_mark_edges_checkbox.isChecked():
                param_dict: RemoveBackgroundDictionary = self.get_background_removal_params_dict()
                message = RemoveBackgroundRequest(
                    source=Modules.CATALOG_HANDLER,
                    destination=Modules.PROCESSING_MODULE,
                    picture=request.frame,
                    param_dict=param_dict)

                response: ProcessedImageResponse = blocking_response_message_await(
                    request_signal=self.qt_signals.processing_module_request,
                    request_message=message,
                    response_signal=self.qt_signals.processing_module_request,
                    response_message_type=ProcessedImageResponse)
                out_image = response.image
                self.video_frame.set_background_image(request.frame)
                self.video_frame.set_front_image(out_image)

                self.uncropped_image = request.frame
                self.cropped_picture = out_image
            else:
                # self.video_frame.set_background_image(request.frame)
                self.video_frame.set_front_image(request.frame)
                self.uncropped_image = request.frame
                self.cropped_picture = None

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
        self.qt_signals.video_thread_request.emit(CameraListMessage())

    def next_picture_button_callback(self):
        print(f"Next button")
        self.image_idx += 1
        self.overlay.crosses = []
        self.request_picture()

    def previous_picture_button_callback(self):
        print(f"Previous button")
        self.image_idx -= 1
        self.overlay.crosses = []
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

    def reset_dropboxes(self):
        self.set_year_dropbox_items()
        year = self.coin_catalog_year_dropbox.currentText()
        self.set_country_dropbox_items(year=year)
        country = self.coin_catalog_country_dropbox.currentText()
        self.set_coin_name_dropbox_items(year=year, country=country)

    def request_picture(self):
        year = self.coin_catalog_year_dropbox.currentText()
        country = self.coin_catalog_country_dropbox.currentText()
        name = self.coin_catalog_name_dropbox.currentText()

        self.active_coin: Coin = self.catalog[year][country][name]

        if len(self.active_coin.pictures.keys()) == 0:
            self.image_idx = 0
            self.current_picture_name = ""
        else:
            self.image_idx %= len(self.active_coin.pictures.keys())
            self.current_picture_name = list(self.active_coin.pictures.keys())[self.image_idx]
            self.qt_signals.catalog_handler_request.emit(PictureRequest(coin=self.active_coin,
                                                                        picture=self.current_picture_name,
                                                                        source=Modules.GALLERY_WINDOW,
                                                                        destination=Modules.CATALOG_HANDLER))

    def save_photo(self):
        request = SavePictureRequest(image_with_background=self.uncropped_image,
                                     cropped_image=self.cropped_picture,
                                     coin=self.active_coin)
        self.qt_signals.catalog_handler_request.emit(request)

    def new_coin_routine(self):
        dialog = NewCoinDialog()
        if dialog.exec():
            print("Dialog closed with Confirm.")
            year = dialog.coin_year_field.text()
            country = dialog.coin_country_field.text()
            name = dialog.coin_name_field.text()

            if year == "" or country == "" or name == "":
                return

            request = NewCoinRequest(coin_year=year,
                                     coin_country=country,
                                     coin_name=name,
                                     source=self.module,
                                     destination=Modules.CATALOG_HANDLER)
            self.qt_signals.catalog_handler_request.emit(request)
        else:
            print("Dialog closed with Cancel.")

    def remove_coin_routine(self):
        request = RemoveCoinRequest(self.active_coin)
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

    def year_dropbox_update_callback(self):
        self.set_country_dropbox_items(year=self.coin_catalog_year_dropbox.currentText())
        self.country_dropbox_update_callback()

    def country_dropbox_update_callback(self):
        self.set_coin_name_dropbox_items(year=self.coin_catalog_year_dropbox.currentText(),
                                         country=self.coin_catalog_country_dropbox.currentText())
        self.name_dropbox_update_callback()

    def name_dropbox_update_callback(self):
        self.update_active_coin()
        self.request_picture()

    def update_active_coin(self):
        year = self.coin_catalog_year_dropbox.currentText()
        country = self.coin_catalog_country_dropbox.currentText()
        name = self.coin_catalog_name_dropbox.currentText()
        self.active_coin = self.catalog[year][country][name]

    def pick_coin_from_dropboxes(self, coin: Coin):
        try:
            _ = self.catalog[coin.year][coin.country][coin.name]
        except:
            return False

        self.set_year_dropbox_items()
        year_id = self.coin_catalog_year_dropbox.findText(coin.year)
        self.coin_catalog_year_dropbox.setCurrentIndex(year_id)

        self.set_country_dropbox_items(year=coin.year)
        country_id = self.coin_catalog_country_dropbox.findText(coin.country)
        self.coin_catalog_country_dropbox.setCurrentIndex(country_id)

        self.set_coin_name_dropbox_items(year=coin.year, country=coin.country)
        name_id = self.coin_catalog_name_dropbox.findText(coin.name)
        self.coin_catalog_name_dropbox.setCurrentIndex(name_id)

        return True

    def reset_coin_vertices(self):
        vertices = []
        request = PictureVerticesUpdateRequest(source=Modules.DRAGGABLE_CROSS_OVERLAY,
                                               destination=Modules.CATALOG_HANDLER,
                                               coin=self.active_coin,
                                               vertices=vertices,
                                               picture_file=self.current_picture_name)
        self.qt_signals.catalog_handler_request.emit(request)

        request = DeleteCroppedPicture(coin=self.active_coin,
                                       picture_file=self.current_picture_name,
                                       source=Modules.DRAGGABLE_CROSS_OVERLAY,
                                       destination=Modules.CATALOG_HANDLER)
        self.qt_signals.catalog_handler_request.emit(request)

        self.request_picture()

    def get_background_removal_params_dict(self) -> dict:
        param_dict: RemoveBackgroundDictionary = RemoveBackgroundDictionary
        current_tab_index = self.tabWidget.currentIndex()
        if self.tabWidget.tabText(current_tab_index) == "Camera":
            param_dict["Blur Kernel"] = self.blur_kernel_slider.value()
            param_dict["Blur Sigma"] = self.blur_sigma_slider.value()
            param_dict["Canny Threshold 1"] = self.canny_thr_1_slider.value()
            param_dict["Canny Threshold 2"] = self.canny_thr_2_slider.value()
            param_dict["Dilate Kernel1"] = self.dilate_1_slider.value()
            param_dict["Dilate Kernel2"] = self.dilate_2_slider.value()
            param_dict["Erode Kernel1"] = self.erode_1_slider.value()
            param_dict["Erode Kernel2"] = self.erode_2_slider.value()
            param_dict["Dilate Iterations"] = self.dilate_iter_slider.value()
            param_dict["Erode Iterations"] = self.erode_iter_slider.value()
        elif self.tabWidget.tabText(current_tab_index) == "Gallery":
            param_dict["Blur Kernel"] = self.gallery_blur_kernel_slider.value()
            param_dict["Blur Sigma"] = 2
            param_dict["Canny Threshold 1"] = 50
            param_dict["Canny Threshold 2"] = 9
            param_dict["Dilate Kernel1"] = self.gallery_dilate_1_slider.value()
            param_dict["Dilate Kernel2"] = self.gallery_dilate_2_slider.value()
            param_dict["Erode Kernel1"] = self.gallery_erode_1_slider.value()
            param_dict["Erode Kernel2"] = self.gallery_erode_2_slider.value()
            param_dict["Dilate Iterations"] = self.gallery_dilate_iter_slider.value()
            param_dict["Erode Iterations"] = self.gallery_erode_iter_slider.value()
        return param_dict

    def handle_redraw_coin_contour_checkbox(self):
        if self.redraw_coin_contour_checkbox.isChecked():
            self.contour_correction_container.setEnabled(True)
            self.video_frame.set_front_image(self.uncropped_image)
        else:
            self.contour_correction_container.setEnabled(False)
            self.video_frame.set_front_image(self.cropped_picture)

    def handle_automatic_contour_detection_checkbox(self):
        if self.automatic_contour_detextion_checkbox.isChecked():
            self.request_background_free_image()
        else:
            self.video_frame.set_front_image(self.uncropped_image)

    def request_background_free_image(self) -> QImage:
        param_dict: RemoveBackgroundDictionary = self.get_background_removal_params_dict()
        message = RemoveBackgroundRequest(
            source=Modules.CATALOG_HANDLER,
            destination=Modules.PROCESSING_MODULE,
            picture=self.uncropped_image,
            param_dict=param_dict)

        response: ProcessedImageResponse = blocking_response_message_await(
            request_signal=self.qt_signals.processing_module_request,
            request_message=message,
            response_signal=self.qt_signals.processing_module_request,
            response_message_type=ProcessedImageResponse)
        self.cropped_picture = response.image
        self.video_frame.set_front_image(self.cropped_picture)

    def save_image_without_background(self):
        if len(self.overlay.crosses) >= 3:
            image_width = self.uncropped_image.width()
            image_height = self.uncropped_image.height()
            window_width = self.video_frame.width()
            window_height = self.video_frame.height()

            vertices = [QPoint(int((cross.x()/window_width)*image_width), int((cross.y()/window_height)*image_height))
                        for cross in self.overlay.crosses]

            message = RemoveBackgroundVerticesRequest(
                source=Modules.CATALOG_HANDLER,
                destination=Modules.PROCESSING_MODULE,
                picture=self.uncropped_image,
                qpoint_vertices=vertices)

            response: ProcessedImageResponse = blocking_response_message_await(
                request_signal=self.qt_signals.processing_module_request,
                request_message=message,
                response_signal=self.qt_signals.processing_module_request,
                response_message_type=ProcessedImageResponse)
            self.cropped_picture = response.image

            request = SaveCroppedPictureRequest(coin=self.active_coin,
                                                picture_name=self.current_picture_name,
                                                image_without_background=self.cropped_picture,
                                                source=Modules.IMAGE_COLLECTOR_WINDOW,
                                                destination=Modules.PROCESSING_MODULE)
            self.qt_signals.catalog_handler_request.emit(request)
        else:
            request = SaveCroppedPictureRequest(coin=self.active_coin,
                                                picture_name=self.current_picture_name,
                                                image_without_background=self.cropped_picture,
                                                source=Modules.IMAGE_COLLECTOR_WINDOW,
                                                destination=Modules.PROCESSING_MODULE)
            self.qt_signals.catalog_handler_request.emit(request)

        self.overlay.crosses = []
        self.request_picture()
