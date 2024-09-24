from PySide6.QtCore import Slot, QTimer
from PySide6.QtWidgets import QMainWindow

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import RequestBase
from core.qt_threading.headers.video_thread.CameraListRequest import CameraListRequest
from core.qt_threading.headers.video_thread.CameraListResponse import CameraListResponse
from core.qt_threading.headers.video_thread.ChangeVideoInput import ChangeVideoInput
from core.qt_threading.headers.video_thread.FrameAvailable import FrameAvailable
from core.ui.ImageFrame import ImageFrame
from core.ui.pyqt6_designer.d_ImageCollector import Ui_w_ImageCollector


class AddNewImageWindow(QMainWindow, Ui_w_ImageCollector):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.qt_signals = CommonSignals()

        # Create an instance of ImageFrame
        self.video_frame = ImageFrame(self.video_frame)

        # Display camera frames
        # self.qt_signals.frame_available.connect(self.replace_video_frame)

        self.qt_signals.video_thread_response.connect(self.receive_request)
        # self.save_photo_button.connect(self.save_photo)
        self.camera_swich_combo_box.currentIndexChanged.connect(self.camera_change_event)
        self.save_photo_button.clicked.connect(self.save_image)

        # Request list of available cameras
        # self.signals.video_thread_request.emit(CameraListRequest())
        QTimer.singleShot(0, self.request_camera_ids)
        # print(f"AddNewImageWindow started: {id(self.signals)}")

    @Slot()
    def closeEvent(self, event):
        super().closeEvent(event)
        # self.signals.video_thread_request.emit(CameraListRequest())
        # self.close()

    @Slot()
    def request_camera_ids(self):
        # self.signals.request_cameras_ids.emit()
        print("request_camera_ids")
        self.qt_signals.video_thread_request.emit(CameraListRequest())
        # self.signals.catalog_handler_request.emit("camera request")

    @Slot()
    def receive_request(self, request: RequestBase):
        print("receive_request")
        if isinstance(request, CameraListResponse):
            # print(f"[AddNewImageWindow]: {request}")
            self.camera_swich_combo_box.addItems(request.body["cameras"])
        elif isinstance(request, FrameAvailable):
            self.video_frame.set_image(request.frame)

    @Slot(list)
    def set_camera_list_combo_box(self, camera_list: list[int]):
        self.camera_swich_combo_box.addItems(camera_list)

    def camera_change_event(self, device_id: int):
        print("camera_change_event")
        self.qt_signals.video_thread_request.emit(ChangeVideoInput(device_id=device_id))

    # @Slot(object)
    # def replace_video_frame(self, frame):
    #     # print("replace_video_frame")
    #     self.video_frame.set_image(frame)
    #     # self.signals.video_thread_request.emit("help")

    @Slot(object)
    def save_image(self):
        # print("replace_video_frame")
        print(f"save picture")
        self.qt_signals.catalog_handler_request.emit("save picture")
        # self.signals.video_thread_request.emit("help")

    # def save_photo(self):
    #     pass
        # self.signals.

