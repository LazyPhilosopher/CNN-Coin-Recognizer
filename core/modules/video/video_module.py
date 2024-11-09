from queue import Queue

import cv2
from PySide6.QtCore import QProcess, QObject, QThread
from PySide6.QtWidgets import QApplication

from core.qt_communication.base import *
from core.qt_communication.messages.video_module.Requests import CameraListRequest
from core.qt_communication.messages.video_module.Responses import *
from core.utilities.helper import cv2_to_qimage


class VideoModule(QObject):

    def __init__(self, device_id=0, size=100):
        super().__init__()
        self.device_id = device_id
        self.size = size
        self.is_running = False

        self.cv2_stream = None
        self.process = None

        self.video_thread = QThread()
        self.temp_thread = QThread()
        self.queue = Queue(maxsize=size)
        self.camera_list: list = []

        qt_signals.video_module_request.connect(self.handle_request)

    def start_process(self):
        """Starts the video capture in a separate QProcess."""
        self.process = QProcess(self)
        self.is_running = True
        self.cv2_stream = cv2.VideoCapture(0)  # Use the first camera (or provide a video file)
        self.cv2_stream.set(cv2.CAP_PROP_FPS, 60)

        self._camera_list_refresh()

        self.moveToThread(self.video_thread)
        self.video_thread.started.connect(self._process_video)
        self.video_thread.start()

    def handle_request(self, request: MessageBase):
        request_handlers = {
            CameraListRequest: self._handle_camera_list_message,
            # ChangeVideoInput: self.handle_change_video_input,
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def _handle_camera_list_message(self, _: CameraListRequest):
        request = CameraListResponse(camera_list=self.camera_list, source=Modules.VIDEO_STREAM)
        qt_signals.video_module_request.emit(request)

    def _process_video(self):
        """Main loop to process the video stream."""
        while self.is_running:
            ret, frame = self.cv2_stream.read()
            if not ret:
                self.reinit_stream(self.device_id)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.queue.put(frame)

            response: MessageBase = FrameAvailable(self._read_frame(), source=Modules.VIDEO_STREAM)
            qt_signals.frame_available.emit(response)
            QApplication.processEvents()

        # Release the capture when done
        self.cv2_stream.release()
        self.video_thread.quit()

    def _read_frame(self) -> QImage:
        return cv2_to_qimage(self.queue.get())

    def _camera_list_refresh(self) -> None:
        index = 0
        id_arr = []
        while True:
            cap = cv2.VideoCapture()
            cap.open(index)
            if not cap.isOpened():
                break
            else:
                id_arr.append(index)
            cap.release()
            index += 1
        self.camera_list = [f"Camera {idx}" for idx in id_arr]
        print(self.camera_list)
        request = CameraListResponse(camera_list=self.camera_list, source=Modules.VIDEO_STREAM)
        qt_signals.video_module_request.emit(request)

    def reinit_stream(self, device_id: int):
        self.cv2_stream.release()
        self.device_id = device_id
        self.cv2_stream = cv2.VideoCapture(self.device_id)
