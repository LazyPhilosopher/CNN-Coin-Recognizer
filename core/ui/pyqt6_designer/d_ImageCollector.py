# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_ImageCollector.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QLabel, QPlainTextEdit, QPushButton, QSizePolicy,
    QWidget)

class Ui_w_ImageCollector(object):
    def setupUi(self, w_ImageCollector):
        if not w_ImageCollector.objectName():
            w_ImageCollector.setObjectName(u"w_ImageCollector")
        w_ImageCollector.resize(606, 401)
        self.video_frame = QFrame(w_ImageCollector)
        self.video_frame.setObjectName(u"video_frame")
        self.video_frame.setGeometry(QRect(10, 10, 381, 381))
        self.video_frame.setFrameShape(QFrame.Panel)
        self.video_frame.setFrameShadow(QFrame.Sunken)
        self.video_frame.setLineWidth(1)
        self.camera_swich_combo_box = QComboBox(w_ImageCollector)
        self.camera_swich_combo_box.setObjectName(u"camera_swich_combo_box")
        self.camera_swich_combo_box.setGeometry(QRect(470, 10, 131, 21))
        self.label_2 = QLabel(w_ImageCollector)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(410, 10, 61, 20))
        self.label_2.setAlignment(Qt.AlignCenter)
        self.line_2 = QFrame(w_ImageCollector)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(400, 10, 16, 381))
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.plainTextEdit = QPlainTextEdit(w_ImageCollector)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(420, 260, 181, 131))
        self.plainTextEdit.setReadOnly(True)
        self.label = QLabel(w_ImageCollector)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(420, 235, 41, 16))
        self.active_coin_combo_box = QComboBox(w_ImageCollector)
        self.active_coin_combo_box.setObjectName(u"active_coin_combo_box")
        self.active_coin_combo_box.setGeometry(QRect(460, 230, 141, 22))
        self.new_coin_button = QPushButton(w_ImageCollector)
        self.new_coin_button.setObjectName(u"new_coin_button")
        self.new_coin_button.setGeometry(QRect(420, 100, 181, 31))
        self.save_photo_button = QPushButton(w_ImageCollector)
        self.save_photo_button.setObjectName(u"save_photo_button")
        self.save_photo_button.setGeometry(QRect(420, 40, 181, 31))
        self.mark_button = QPushButton(w_ImageCollector)
        self.mark_button.setObjectName(u"mark_button")
        self.mark_button.setGeometry(QRect(420, 140, 181, 31))
        self.checkBox = QCheckBox(w_ImageCollector)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(420, 70, 171, 31))

        self.retranslateUi(w_ImageCollector)

        QMetaObject.connectSlotsByName(w_ImageCollector)
    # setupUi

    def retranslateUi(self, w_ImageCollector):
        w_ImageCollector.setWindowTitle(QCoreApplication.translate("w_ImageCollector", u"Form", None))
        self.label_2.setText(QCoreApplication.translate("w_ImageCollector", u"Camera", None))
        self.plainTextEdit.setPlainText("")
        self.label.setText(QCoreApplication.translate("w_ImageCollector", u"Coin", None))
        self.new_coin_button.setText(QCoreApplication.translate("w_ImageCollector", u"New Coin", None))
        self.save_photo_button.setText(QCoreApplication.translate("w_ImageCollector", u"New Photo", None))
        self.mark_button.setText(QCoreApplication.translate("w_ImageCollector", u"Mark Coin Edges", None))
        self.checkBox.setText(QCoreApplication.translate("w_ImageCollector", u"auto-mark coin edges", None))
    # retranslateUi

