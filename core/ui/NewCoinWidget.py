import sys
import typing
import uuid

import numpy
import os
from PySide6.QtCore import QRectF, Qt, QEventLoop
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QFrame, QWidget, QLabel, QComboBox, QPushButton
from PySide6.QtGui import QPixmap, QScreen, QImage

from core.ui.pyqt6_designer.d_NewCoinWidget import Ui_NewCoinWidget
from core.threading.signals import ThreadingSignals


loop = QEventLoop()


class NewCoinWidget(QWidget, Ui_NewCoinWidget):
    def __init__(self, signals: ThreadingSignals):
        super().__init__()
        self.setupUi(self)
        self.signals = signals
        self.ok_button.pressed.connect(self.send_new_coin_signal)
        self.signals.info_signal.connect(self.acknowledgement_handler)
        self.module_sender_uuid = uuid.uuid4()

    def initUI(self):
        self.setWindowTitle("Create new Coin")
        self.setGeometry(200, 200, 400, 300)

    def acknowledgement_handler(self, status, data):
        if data["receiver_id"] == self.module_sender_uuid:
            if status:
                print("Data processed successfully for identifier:", data)
                loop.quit()
                self.close()
            else:
                print("Data processing failed for identifier:", data["message"])

    def send_new_coin_signal(self):
        coin_dict = {'sender_id': self.module_sender_uuid,
                     'name': self.coin_name_field.text(),
                     'year': self.coin_year_field.text(),
                     'country': self.coin_country_field.text(),
                     'weight': self.coin_weight_field.text(),
                     'content': self.coin_content_field.text()}
        self.signals.new_coin_created.emit(coin_dict)

        loop.exec_()
        #self.close()
