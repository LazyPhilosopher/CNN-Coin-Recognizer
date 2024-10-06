from queue import Queue

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QObject, QThread
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.messages.MessageBase import MessageBase, Modules
from core.qt_threading.messages.video_thread.Requests import CameraListMessage, ChangeVideoInput, FrameAvailable
from core.qt_threading.messages.video_thread.Responses import CameraListResponse


class VideoStream(QObject):

    def __init__(self, device_id=0, size=100):
        super().__init__()
        self.device_id = device_id
        self.size = size
        self.is_running = False

        self.cv2_stream = None
        self.process = None
        self.qt_signals = CommonSignals()
        self.video_thread = QThread()
        self.temp_thread = QThread()
        self.queue = Queue(maxsize=size)
        self.camera_list: list = []

        self.qt_signals.video_thread_request.connect(self.handle_request)

    def start_process(self):
        """Starts the video capture in a separate QProcess."""
        self.process = QProcess(self)
        self.is_running = True
        self.cv2_stream = cv2.VideoCapture(0)  # Use the first camera (or provide a video file)
        self.cv2_stream.set(cv2.CAP_PROP_FPS, 30)

        self._camera_list_refresh()

        # Move the processing function to the background
        # self.moveToThread(self.temp_thread)
        # self.temp_thread.started.connect(self.return_camera_list)
        # self.temp_thread.start()

        self.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.process_video)
        self.video_thread.start()
        # print(f"VideoStream started: {id(self.signals)}")

    def reinit_stream(self, device_id: int):
        self.cv2_stream.release()
        self.device_id = device_id
        self.cv2_stream = cv2.VideoCapture(self.device_id)

    def handle_request(self, request: MessageBase):
        request_handlers = {
            CameraListMessage: self.handle_camera_list_message,
            ChangeVideoInput: self.handle_change_video_input,
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            handler(request)

    def handle_camera_list_message(self, _: CameraListMessage):
        self.moveToThread(self.temp_thread)
        self.temp_thread.started.connect(self._camera_list_refresh)
        self.temp_thread.start()

    def handle_change_video_input(self, request: ChangeVideoInput):
        if request.device_id != self.device_id:
            self.reinit_stream(request.device_id)

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

        request = CameraListResponse(camera_list=self.camera_list, source=Modules.VIDEO_STREAM)
        self.qt_signals.video_thread_request.emit(request)

    def process_video(self):
        """Main loop to process the video stream."""
        while self.is_running:
            ret, frame = self.cv2_stream.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.queue.put(frame)

            response: MessageBase = FrameAvailable(self.read_frame(), source=Modules.VIDEO_STREAM)
            self.qt_signals.frame_available.emit(response)
            QApplication.processEvents()

        # Release the capture when done
        self.cv2_stream.release()
        self.video_thread.quit()

    def read_frame(self) -> QImage:
        return self.NumpyToQImage(self.queue.get())

    def NumpyToQImage(self, image_data) -> QImage:
        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 2:  # Grayscale image
                height, width = image_data.shape
                bytes_per_line = width
                q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif image_data.ndim == 3:  # Color image
                height, width, channels = image_data.shape
                if channels == 3:  # RGB image
                    bytes_per_line = channels * width
                    q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                elif channels == 4:  # RGBA image
                    bytes_per_line = channels * width
                    q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
                else:
                    raise ValueError("Unsupported number of channels")
            else:
                raise ValueError("Unsupported ndarray shape")

            return q_image
        else:
            return image_data

