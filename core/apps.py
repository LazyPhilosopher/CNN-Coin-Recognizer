from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from cv2 import cornerSubPix
from gui.frames.ImageCaptureFrame import ImageCaptureFrame
from utils.video import VideoStream, QPixmap, QImage
from PyQt5.QtCore import Qt
from core.signals import CustomSignals
from utils.CV2Module import CV2Module

custom_signals = CustomSignals()


class ImageRefreshModule:
    def __init__(self, frame, video_stream) -> None:
        self.frame = frame 
        self.video_stream = video_stream
        pass
    
    def refresh(self):
        while True:
            if self.video_stream.check_queue():
                frame = self.video_stream.read()
                self.frame.set_picture(frame)
                pass

    def set_picture(self, picture):
        height, width, channel = picture.shape
        bytes_per_line = 3 * width
        qimage = QImage(picture.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.frame.set_picture(picture)
        # create qpixmap from qimage
        #pixmap = QPixmap.fromImage(qimage)
        #pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation) 
        #self.frame.sized_pixmap.setPixmap(pixmap)  # Update the pixmap in the QGraphicsPixmapItem
        #self.frame.scene.update()
                

class ImageCaptureApp:
    def __init__(self) -> None:
        app = QApplication([])
        
        self.vs = VideoStream(custom_signals.frame_available, device=0)
        self.main_window = ImageCaptureFrame(custom_signals=custom_signals, list_camera_ids_list=self.vs.get_camera_ids_list())
        self.vs.start()
        self.image_refresh_module = ImageRefreshModule(self.main_window, self.vs)
        self.cv2_module = CV2Module()
        
        custom_signals.frame_available.connect(self.frame_update)
        custom_signals.camera_reinit_signal.connect(self.video_stream_reinit)
        
        self.main_window.show()
        app.exec_()          
        
    def frame_update(self, frame):
        (corners, _, _) = self.cv2_module.detect_markers_on_frame(frame)
        frame = self.cv2_module.colorize_markers_on_frame(frame=frame, corners=corners)
        self.image_refresh_module.set_picture(frame)
        
    def video_stream_reinit(self, camera_id = 0):
        self.vs.stop()
        self.vs = VideoStream(custom_signals.frame_available, device=camera_id).start()
        
    
        
        