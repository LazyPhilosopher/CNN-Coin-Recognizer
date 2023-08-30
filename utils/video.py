import cv2
from queue import Queue
from threading import Thread
import time

from PyQt5.QtGui import QColor, QPainterPath, QPen, QPixmap, QImage
from PyQt5.QtCore import QObject


class VideoStream(QObject):  # Derive from QObject to use signals

    def __init__(self, signal, device=0, size=100):
        super().__init__()
        self.stream = cv2.VideoCapture(device)
        self.stream.set(cv2.CAP_PROP_FPS, 10)
        self.stopped = False
        self.queue = Queue(maxsize=size)
        self.refresh_signal = signal         

    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        while self.stopped is False:
            if not self.queue.full():
                (grabbed, frame) = self.stream.read()

            if not grabbed:
                self.stop()
                return
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.queue.put(frame)
            # self.queue.task_done()  # Mark the frame as processed
            self.refresh_signal.emit(self.read())  # Emit the frame through the signal
            # print(f"Size of queue: {self.queue.qsize()}")

    def read(self):
        return self.queue.get()

    def check_queue(self):
        return self.queue.qsize() > 0

    def stop(self):
        self.stopped = True
        self.stream.release()
        
    def get_camera_ids_list(self):
        index = 0
        arr = []
        while True:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.read()[0]:
                    break
                else:
                    arr.append(index)
                cap.release()
                index += 1
            except:
                break
        return arr
        