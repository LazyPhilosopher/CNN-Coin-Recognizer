import os
from pathlib import Path

from PySide6.QtCore import QPoint, Slot
from PySide6.QtGui import QImage, QIcon
from PySide6.QtWidgets import QMainWindow, QLabel

from core.gui.modules.DraggableCrossesOverlay import DraggableCrossesOverlay
from core.gui.modules.ImageFrame import ImageFrame
from core.gui.modules.NewCoinDialog import NewCoinDialog
from core.gui.pyqt6_designer.d_ImageCollector import Ui_ImageCollector
from core.qt_communication.base import *
from core.qt_communication.messages.processing_module.Requests import RemoveBackgroundRequest
from core.qt_communication.messages.processing_module.Responses import ProcessedImageResponse
from core.qt_communication.messages.video_module.Requests import CameraListRequest, ChangeVideoInput
from core.qt_communication.messages.video_module.Responses import CameraListResponse, FrameAvailable
from core.utilities.helper import parse_directory_into_dictionary, create_coin_directory, get_files, resource_path, get_tab_index_by_label, crop_vertices_mask_from_image

catalog_dir = Path("coin_catalog")


class ImageCollector(QMainWindow, Ui_ImageCollector):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.qt_signals = CommonSignals()
        self.module: Modules = Modules.IMAGE_COLLECTOR_WINDOW

        self.video_frame = ImageFrame(self.video_frame)
        self.image_label = QLabel(self.video_frame)
        self.image_label.setGeometry(0, 0, self.video_frame.width(), self.video_frame.height())
        self.image_label.setScaledContents(True)

        self.setWindowIcon(QIcon(resource_path("core/gui/images/camera.png")))
        self.coin_catalog = None
        self.image_idx: int = 0

        self.overlay = DraggableCrossesOverlay(self.video_frame)
        self.overlay.setGeometry(self.video_frame.rect())

        self._init()  # Connect timeout signal

        QTimer.singleShot(0, self._request_camera_ids)

    def _init(self):
        self.new_coin_button.clicked.connect(self._new_coin_routine)
        self.coin_catalog_country_dropbox.currentIndexChanged.connect(self.country_dropbox_update_callback)
        self.coin_catalog_name_dropbox.currentIndexChanged.connect(self.coin_name_dropbox_update_callback)
        self.save_photo_button.clicked.connect(self.save_photo_button_callback)
        self.tabWidget.tabBarClicked.connect(self._tab_bar_click_callback)
        self.overlay.crosses_changed.connect(self._overlay_click_callback)

        # == Gallery Tab Interface ==
        self.next_gallery_photo_button.clicked.connect(lambda: self.picture_next_prev_button_callback(1))
        self.previous_gallery_photo_button.clicked.connect(lambda: self.picture_next_prev_button_callback(-1))
        self.delete_background_button.clicked.connect(self.delete_background_button_callback)
        self.reset_button.clicked.connect(self.reset_button_callback)
        self.reset_vertices_button.clicked.connect(self.reset_vertices_button_callback)
        self.crop_with_vertices_button.clicked.connect(self.crop_image_wit_vertices)
        self.gallery_save_button.clicked.connect(self.gallery_save_button_routine)

        # Outer module incoming requests
        # self.qt_signals.catalog_handler_response.connect(self.handle_request)
        self.qt_signals.video_module_request.connect(self.handle_request)
        self.qt_signals.frame_received.connect(self.handle_request)

        # self.qt_signals.catalog_handler_request.emit(CatalogDictRequest(source=Modules.GALLERY_WINDOW))
        self.camera_swich_combo_box.currentIndexChanged.connect(self._camera_change_request)

        self.coin_catalog = parse_directory_into_dictionary(catalog_dir)
        self._refresh_coin_dropbox()

    def handle_request(self, request: MessageBase):
        request_handlers = {
            # CatalogDictResponse: self.handle_catalog_dict_response,
            CameraListResponse: self._handle_camera_list_response,
            # PictureResponse: self.handle_picture_response,
            FrameAvailable: self._handle_frame_available_request
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def _handle_camera_list_response(self, request: CameraListResponse):
        self.camera_swich_combo_box.clear()
        self.camera_swich_combo_box.addItems(request.cameras)

    def _handle_frame_available_request(self, request: FrameAvailable):
        current_tab_index = self.tabWidget.currentIndex()
        if self.tabWidget.tabText(current_tab_index) == "Camera":
            if self.auto_background_deletion_checkbox.isChecked():

                self.qt_signals.frame_received.disconnect(self.handle_request)
                message = RemoveBackgroundRequest(
                    source=Modules.CATALOG_HANDLER,
                    destination=Modules.PROCESSING_MODULE,
                    picture=request.frame)

                response: ProcessedImageResponse = blocking_response_message_await(
                    request_signal=self.qt_signals.processing_module_request,
                    request_message=message,
                    response_signal=self.qt_signals.processing_module_request,
                    response_message_type=ProcessedImageResponse)
                self.qt_signals.frame_received.connect(self.handle_request)

                if response is None:
                    # response wasn't received in time
                    uncropped_image = QImage("core/gui/pictures/default_image.jpg")
                    self.video_frame.set_image(cropped_image=None, uncropped_image=uncropped_image)
                else:
                    self.video_frame.set_image(cropped_image=response.image, uncropped_image=request.frame)
            else:
                self.video_frame.set_image(cropped_image=None, uncropped_image=request.frame)

    @Slot()
    def _request_camera_ids(self):
        self.qt_signals.video_module_request.emit(CameraListRequest())

    def _tab_bar_click_callback(self, index):
        self.tabWidget.setCurrentIndex(index)
        self.overlay.reset_crosses()

        clicked_tab_text = self.tabWidget.tabText(index)
        if clicked_tab_text == "Gallery":
            self.overlay.setEnabled(True)
            self.picture_next_prev_button_callback(0)
        elif clicked_tab_text == "Camera":
            self.overlay.setEnabled(False)

    def _overlay_click_callback(self):
        cross_cnt = len(self.overlay.crosses)
        self.reset_vertices_button.setEnabled(cross_cnt > 0)
        self.crop_with_vertices_button.setEnabled(cross_cnt > 2)

    def _camera_change_request(self):
        camera_id: int = self.camera_swich_combo_box.currentIndex()
        self.qt_signals.video_module_request.emit(ChangeVideoInput(device_id=camera_id))


    # == Coin Dropbox Routine ==

    def _clear_coin_dropboxes(self):
        self.coin_catalog_country_dropbox.clear()
        self.coin_catalog_name_dropbox.clear()
        self.coin_catalog_year_dropbox.clear()

    def _refresh_coin_dropbox(self):
        self._clear_coin_dropboxes()

        try:
            self.set_country_dropbox_items(None)
            self.set_coin_name_dropbox_items(None)
            self.set_coin_year_dropbox_items(None)
            self.save_photo_button.setEnabled(True)
        except:
            self._empty_and_disable_coin_dropboxes()
            self.save_photo_button.setEnabled(False)

    def _set_coin_dropbox_items(self, country: str, coin_name: str, coin_year: str):
        self.set_country_dropbox_items(country)
        self.set_coin_name_dropbox_items(coin_name)
        self.set_coin_year_dropbox_items(coin_year)

    def _empty_and_disable_coin_dropboxes(self):
        self._clear_coin_dropboxes()
        self.coin_catalog_country_dropbox.addItems(["None Defined"])
        self.coin_catalog_name_dropbox.addItems(["None Defined"])
        self.coin_catalog_year_dropbox.addItems(["None Defined"])

        self.coin_catalog_country_dropbox.setEnabled(False)
        self.coin_catalog_name_dropbox.setEnabled(False)
        self.coin_catalog_year_dropbox.setEnabled(False)

        self.save_photo_button.setEnabled(False)

    def _enable_coin_dropboxes(self):
        self._clear_coin_dropboxes()
        self.coin_catalog_country_dropbox.setEnabled(True)
        self.coin_catalog_name_dropbox.setEnabled(True)
        self.coin_catalog_year_dropbox.setEnabled(True)

        self.save_photo_button.setEnabled(True)

    def get_dropbox_values(self):
        country = self.coin_catalog_country_dropbox.currentText()
        coin_name = self.coin_catalog_name_dropbox.currentText()
        year = self.coin_catalog_year_dropbox.currentText()
        return country, coin_name, year

    def _new_coin_routine(self):
        dialog = NewCoinDialog()
        if dialog.exec():
            print("Dialog closed with Confirm.")
            year = dialog.coin_year_field.text()
            country = dialog.coin_country_field.text()
            name = dialog.coin_name_field.text()

            if year == "" or country == "" or name == "":
                return

            create_coin_directory(catalog_dir, country, name, year)
            self.coin_catalog = parse_directory_into_dictionary(catalog_dir)
            self._refresh_coin_dropbox()
            self._set_coin_dropbox_items(country, name, year)

    def country_dropbox_update_callback(self):
        self.set_country_dropbox_items(self.coin_catalog_country_dropbox.currentText())

    def coin_name_dropbox_update_callback(self):
        self.set_coin_name_dropbox_items(self.coin_catalog_name_dropbox.currentText())

    def set_country_dropbox_items(self, country: str = None):
        self.coin_catalog_country_dropbox.blockSignals(True)

        self.coin_catalog_country_dropbox.clear()

        if len(list(self.coin_catalog.keys())) > 0:
            self.coin_catalog_country_dropbox.addItems(list(self.coin_catalog.keys()))
            if country is not None:
                self.coin_catalog_country_dropbox.setCurrentText(country)
            else:
                self.coin_catalog_country_dropbox.setCurrentText(list(self.coin_catalog.keys())[0])
            self.coin_catalog_country_dropbox.setEnabled(True)
        else:
            self._empty_and_disable_coin_dropboxes()
            return

        self.set_coin_name_dropbox_items(None)
        self.coin_catalog_country_dropbox.blockSignals(False)

    def set_coin_name_dropbox_items(self, coin_name: str = None):
        self.coin_catalog_name_dropbox.blockSignals(True)

        self.coin_catalog_name_dropbox.clear()
        country = self.coin_catalog_country_dropbox.currentText()

        if country == "None Defined":
            self._empty_and_disable_coin_dropboxes()
            return

        if len(list(self.coin_catalog[country].keys())) > 0:
            self.coin_catalog_name_dropbox.addItems(list(self.coin_catalog[country].keys()))
            self.coin_catalog_name_dropbox.setEnabled(True)
        else:
            self._empty_and_disable_coin_dropboxes()
            return

        if coin_name is not None and coin_name in self.coin_catalog[country].keys():
            self.coin_catalog_name_dropbox.setCurrentText(coin_name)
        else:
            self.coin_catalog_name_dropbox.setCurrentText(list(self.coin_catalog[country].keys())[0])

        self.set_coin_year_dropbox_items(None)
        self.coin_catalog_name_dropbox.blockSignals(False)

    def set_coin_year_dropbox_items(self, year: str | int = None):
        self.coin_catalog_year_dropbox.blockSignals(True)

        self.coin_catalog_year_dropbox.clear()
        country = self.coin_catalog_country_dropbox.currentText()
        coin_name = self.coin_catalog_name_dropbox.currentText()

        if country == "None Defined" or coin_name == "None Defined":
            self._empty_and_disable_coin_dropboxes()
            return

        if len(list(self.coin_catalog[country][coin_name].keys())) > 0:
            self.coin_catalog_year_dropbox.addItems(list(self.coin_catalog[country][coin_name].keys()))
            self.coin_catalog_year_dropbox.setEnabled(True)
        else:
            self._empty_and_disable_coin_dropboxes()
            return

        if year is not None and self.coin_catalog[country][coin_name].keys():
            year = str(year)
            self.coin_catalog_year_dropbox.setCurrentText(year)
        else:
            self.coin_catalog_year_dropbox.setCurrentText(list(self.coin_catalog[country][coin_name].keys())[0])

        self.coin_catalog_year_dropbox.blockSignals(False)
        self.picture_next_prev_button_callback(0)

    # == Coin Image Routine ==

    def save_photo_button_callback(self):
        country, coin_name, year = self.get_dropbox_values()

        idx = len(get_files(Path(catalog_dir / country / coin_name / year / "uncropped")))

        self.video_frame.uncropped_image.save(
            os.path.join(catalog_dir, country, coin_name, year, "uncropped", str(idx)+'.png'), "PNG")
        if self.video_frame.cropped_image is not None:
            self.video_frame.cropped_image.save(
                os.path.join(catalog_dir, country, coin_name, year, "cropped", str(idx)+'.png'), "PNG")

    # == Gallery Tab Routine ==

    def picture_next_prev_button_callback(self, step: int):
        clicked_tab_text = self.tabWidget.tabText(self.tabWidget.currentIndex())
        if clicked_tab_text != "Gallery":
            return

        self.image_idx += step
        country, coin_name, year = self.get_dropbox_values()

        uncropped_dir_path = Path(catalog_dir / country / coin_name / year / "uncropped")
        cropped_dir_path = Path(catalog_dir / country / coin_name / year / "cropped")

        uncropped_images = get_files(uncropped_dir_path)
        cropped_images = get_files(cropped_dir_path)

        if len(uncropped_images) == 0:
            self.video_frame.set_image(uncropped_image=QImage(Path("core", "gui", "pictures", "default_image.jpg")), cropped_image=None)
            idx = get_tab_index_by_label(self.tabWidget, "Gallery")
            self.tabWidget.widget(idx).setEnabled(False)

        else:
            idx = get_tab_index_by_label(self.tabWidget, "Gallery")
            self.tabWidget.widget(idx).setEnabled(True)

            self.image_idx %= len(uncropped_images)

            image = uncropped_images[self.image_idx]
            if any(path.name == image.name for path in cropped_images):
                uncropped_image = QImage(uncropped_images[self.image_idx])
                cropped_image = QImage(cropped_images[self.image_idx])
                self.video_frame.set_image(cropped_image=cropped_image, uncropped_image=uncropped_image)
            else:
                uncropped_image = QImage(uncropped_images[self.image_idx])
                self.video_frame.set_image(cropped_image=None, uncropped_image=uncropped_image)

    def delete_background_button_callback(self):
        uncropped_image: QImage = self.video_frame.uncropped_image
        if self.video_frame.cropped_image is not None:
            cropped_image = self.video_frame.cropped_image
        else:
            cropped_image = uncropped_image
        message = RemoveBackgroundRequest(
            source=Modules.CATALOG_HANDLER,
            destination=Modules.PROCESSING_MODULE,
            picture=cropped_image)

        response: ProcessedImageResponse = blocking_response_message_await(
            request_signal=self.qt_signals.processing_module_request,
            request_message=message,
            response_signal=self.qt_signals.processing_module_request,
            response_message_type=ProcessedImageResponse)

        cropped_image = response.image
        self.video_frame.set_image(uncropped_image=uncropped_image, cropped_image=cropped_image)

    def reset_button_callback(self):
        uncropped_image: QImage = self.video_frame.uncropped_image
        self.video_frame.set_image(uncropped_image=uncropped_image, cropped_image=None)

    def reset_vertices_button_callback(self):
        self.overlay.reset_crosses()
        self.overlay.show()

    def crop_image_wit_vertices(self):
        if len(self.overlay.crosses) >= 3:
            image: QImage = self.video_frame.front_image_label.pixmap().toImage()
            image_width = image.width()
            image_height = image.height()
            window_width = self.video_frame.width()
            window_height = self.video_frame.height()

            vertices = [QPoint(int((cross.x()/window_width)*image_width), int((cross.y()/window_height)*image_height))
                        for cross in self.overlay.crosses]

            cropped_image: QImage = crop_vertices_mask_from_image(image, vertices)
            self.video_frame.set_image(uncropped_image=self.video_frame.uncropped_image, cropped_image=cropped_image)

        self.overlay.reset_crosses()

    def gallery_save_button_routine(self):
        country, coin_name, year = self.get_dropbox_values()
        uncropped_dir_path = Path(catalog_dir / country / coin_name / year / "uncropped")
        cropped_dir_path = Path(catalog_dir / country / coin_name / year / "cropped")

        uncropped_images = get_files(uncropped_dir_path)
        self.image_idx %= len(uncropped_images)
        image_filename: str = uncropped_images[self.image_idx].parts[-1]

        image = self.video_frame.front_image_label.pixmap().toImage()
        image.save(str(Path(cropped_dir_path / image_filename)), "PNG")
