from PySide6.QtWidgets import QApplication
# from core.signals import CustomSignals
# from utils.CV2Module import CV2Module

from core.catalog.ImageCollector import ImageCollector
from core.video.video import VideoStream
# from utils.CV2Module import CV2Module
from core.threading.signals import ThreadingSignals


signals = ThreadingSignals()


# class ImageRefreshModule:
#     def __init__(self, picture: QLabel, video_stream: VideoStream) -> None:
#         self.picture = picture
#         self.video_stream = video_stream
#         pass


class ImageCaptureApp:
    def __init__(self) -> None:
        app = QApplication([])

        self.main_window = ImageCollector(signals=signals)
        self.video_stream = VideoStream(signals.frame_available, device=0)
        self.main_window.set_camera_combo_box(self.video_stream.available_camera_ids)

        # self.image_refresh_module = ImageRefreshModule(self.main_window.image_label, self.video_stream)
        # self.cv2_module = cv2Module()

        signals.frame_available.connect(self.frame_update)
        signals.camera_reinit_signal.connect(self.video_stream_reinit)
        self.video_stream.start()
        self.main_window.show()
        app.exec_()

    def frame_update(self, frame):
        # (corners, _, _) = self.cv2_module.detect_markers_on_frame(frame)
        # frame = self.cv2_module.colorize_markers_on_frame(frame=frame, corners=corners)
        self.main_window.set_image(frame)

    def video_stream_reinit(self, camera_id=0):
        self.video_stream.stop()
        self.video_stream = VideoStream(signals.frame_available, device=camera_id).start()
