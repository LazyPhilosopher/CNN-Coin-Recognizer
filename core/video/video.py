import time

import cv2
from queue import Queue
from PySide6.QtCore import QProcess, QObject, Slot, QThread
from PySide6.QtWidgets import QApplication

from core.qt_threading.common_signals import CommonSignals
from core.qt_threading.headers.RequestBase import RequestBase, Modules
from core.qt_threading.headers.video_thread.CameraListRequest import CameraListRequest
from core.qt_threading.headers.video_thread.CameraListResponse import CameraListResponse
from core.qt_threading.headers.video_thread.ChangeVideoInput import ChangeVideoInput
from core.qt_threading.headers.video_thread.FrameAvailable import FrameAvailable


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

        self.qt_signals.video_thread_request.connect(self.receive_request)

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

    @Slot()
    def receive_request(self, request: RequestBase):
        print(f"VideoThread request: {request}")
        match request:
            case CameraListRequest():
                # time.sleep(3)
                self.moveToThread(self.temp_thread)
                self.temp_thread.started.connect(self._camera_list_refresh)
                self.temp_thread.start()

                # print("VideoThread CameraListResponse")
            case ChangeVideoInput(body={"device_id": device_id}):
                pass
                # print(f"ChangeVideoInput: {type(request)}")
                if device_id != self.device_id:
                    self.reinit_stream(device_id)
            case _:
                print(f"Unhandled request type: {type(request)}")

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
        self.qt_signals.video_thread_response.emit(request)

    # def return_camera_list(self):
    #     self._get_camera_ids_list()
    #     self.qt_signals.video_thread_response.emit(
    #         CameraListResponse(self.camera_list)
    #     )

    def process_video(self):
        """Main loop to process the video stream."""
        while self.is_running:
            ret, frame = self.cv2_stream.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.queue.put(frame)

            response: RequestBase = FrameAvailable(self.read_frame(), source=Modules.VIDEO_STREAM)
            self.qt_signals.video_thread_response.emit(response)
            # print("self.signals.frame_available.emit(self.read()) ")
            # print(f"Size of queue: {self.queue.qsize()}")
            QApplication.processEvents()

        # Release the capture when done
        self.cv2_stream.release()
        self.video_thread.quit()

    def read_frame(self):
        return self.queue.get()

    # def stop(self):
    #     """Stop the video stream and terminate the process."""
    #     self.is_running = False
    #     self.video_thread.quit()
    #     self.video_thread.wait()
