from PyQt5.QtCore import pyqtSignal, QObject


class CustomSignals(QObject):
    frame_available = pyqtSignal(object)
    save_picture = pyqtSignal(object)
    