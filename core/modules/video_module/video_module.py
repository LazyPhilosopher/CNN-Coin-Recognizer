import time
import ctypes
from ctypes import wintypes

import cv2
from PySide6.QtCore import (QObject, QThread, QProcess, QTimer,
                            QAbstractNativeEventFilter, QCoreApplication)
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import QApplication

from core.qt_communication.base import *
from core.qt_communication.messages.video_module.Requests import CameraListRequest, ChangeVideoInput
from core.qt_communication.messages.video_module.Responses import *
from core.utilities.core_utils import suppress_stderr
from core.utilities.helper import cv2_to_qimage


class VideoModule(QObject):

    def __init__(self, device_id=0):
        super().__init__()
        self.device_id = device_id
        self.qt_signals = CommonSignals()
        self.is_running = False

        self.cv2_stream = None
        self.process = None
        self.max_frame_retry_allowed = 300
        self.failed_frames_cnt = 0
        self.camera_list: list = []

        self.video_stream_timer = QTimer()
        self.camera_refresh_timer = QTimer()

        self.qt_signals.video_module_request.connect(self.handle_request)

        self.media_devices = QMediaDevices()
        self.media_devices.videoInputsChanged.connect(self._camera_list_refresh)

    def start_process(self):
        """Starts the video capture process in a separate thread."""
        self.is_running = True

        self.camera_refresh_timer.timeout.connect(self._check_camera_presence)
        self.camera_refresh_timer.setInterval(1000)

        self.video_stream_timer.timeout.connect(self._read_video_stream_frame)
        self.video_stream_timer.setInterval(int(1000/30))

        if self._camera_list_refresh():
            self.start_video_stream_thread()
        else:
            self.camera_refresh_timer.start()

    def handle_request(self, request: MessageBase):
        """Handles incoming messages."""
        request_handlers = {
            CameraListRequest: self._handle_camera_list_message,
            ChangeVideoInput: self._handle_video_input_change
            # Additional request handlers...
        }
        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def _handle_camera_list_message(self, _: CameraListRequest = None):
        request = CameraListResponse(camera_list=self.camera_list, source=Modules.VIDEO_STREAM)
        self.qt_signals.video_module_request.emit(request)

    def _handle_video_input_change(self, request: ChangeVideoInput):
        self.reinit_stream(device_id=request.device_id)

    def _read_video_stream_frame(self):
        with suppress_stderr():
            ret, frame = self.cv2_stream.read()

        if not ret:
            self.failed_frames_cnt += 1
            print(f"Failed frames: {self.failed_frames_cnt} / {self.max_frame_retry_allowed}")

            if self.failed_frames_cnt > self.max_frame_retry_allowed:
                self._switch_to_camera_refresh()
                return

            self.reinit_stream(self.device_id)
            response: MessageBase = FrameNotAvailable(source=Modules.VIDEO_STREAM)

        else:
            self.failed_frames_cnt = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2_to_qimage(frame)
            response: MessageBase = FrameAvailable(frame, source=Modules.VIDEO_STREAM)

        self.qt_signals.frame_received.emit(response)
        QApplication.processEvents()

    def _switch_to_camera_refresh(self):
        """Stops the video stream and switches to refreshing camera list."""
        print("Switching to camera refresh mode.")
        self.stop_video_stream_thread()
        self.camera_refresh_timer.start()

    def _camera_list_refresh(self) -> bool:
        """Refreshes the list of available cameras and returns True if at least one is found."""
        index = 0
        id_arr = []

        while True:
            self.qt_signals.video_module_request.emit(CameraListResponse(camera_list=[], source=Modules.VIDEO_STREAM))
            with suppress_stderr():
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            else:
                id_arr.append(index)
            cap.release()
            index += 1

        self.camera_list = [f"Camera {idx}" for idx in id_arr]
        request = CameraListResponse(camera_list=self.camera_list, source=Modules.VIDEO_STREAM)
        self.qt_signals.video_module_request.emit(request)

        return bool(self.camera_list)

    def reinit_stream(self, device_id: int):
        """Reinitializes the camera stream."""
        self.cv2_stream.release()
        self.device_id = device_id
        with suppress_stderr():
            self.cv2_stream = cv2.VideoCapture(self.device_id)

    def _check_camera_presence(self):
        if not self._camera_list_refresh():
            return

        self.camera_refresh_timer.stop()
        self.start_video_stream_thread()

    def start_video_stream_thread(self, device_id: int = None):
        if device_id:
            self.device_id = device_id

        if self.device_id >= len(self.camera_list):
            self.device_id = 0

        with suppress_stderr():
            self.cv2_stream = cv2.VideoCapture(self.device_id)
        self.cv2_stream.set(cv2.CAP_PROP_FPS, 60)

        self.video_stream_timer.start()

    def stop_video_stream_thread(self):
        if self.cv2_stream:
            self.cv2_stream.release()
        if self.video_stream_thread.isRunning():
            self.video_stream_thread.quit()
            self.video_stream_thread.wait()
            self.video_stream_thread = None
