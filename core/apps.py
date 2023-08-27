from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from gui.frames.ImageCaptureFrame import ImageCaptureFrame
from utils.video import VideoStream
import time
import threading
from PyQt5.QtGui import QColor, QPainterPath, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject


class CustomSignals(QObject):
    frame_available = pyqtSignal(object)


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
                # time.sleep(.1)

    def set_picture(self, picture):
        height, width, channel = picture.shape
        bytes_per_line = 3 * width
        qimage = QImage(picture.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # create qpixmap from qimage
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation) 
        self.frame.pixmap_item.setPixmap(pixmap)  # Update the pixmap in the QGraphicsPixmapItem
        self.frame.scene.update()
                

class ImageCaptureApp:
    def __init__(self) -> None:
        app = QApplication([])
        self.main_window = ImageCaptureFrame()
        self.signals = CustomSignals()
        self.vs = VideoStream(self.signals.frame_available, device=1).start()
        self.image_refresh_module = ImageRefreshModule(self.main_window, self.vs)
        
        self.signals.frame_available.connect(self.image_refresh_module.set_picture)
        
        
        self.main_window.show()
        
        # Start the refresh loop in a separate thread
        # refresh_thread = threading.Thread(target=self.vs.update)
        # refresh_thread.daemon = True  # This will allow the thread to exit when the main thread exits
        # refresh_thread.start()

        app.exec_()          
        
