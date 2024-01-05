# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_ImageCollector.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
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
    QTabWidget, QWidget)

class Ui_w_ImageCollector(object):
    def setupUi(self, w_ImageCollector):
        if not w_ImageCollector.objectName():
            w_ImageCollector.setObjectName(u"w_ImageCollector")
        w_ImageCollector.resize(661, 401)
        self.video_frame = QFrame(w_ImageCollector)
        self.video_frame.setObjectName(u"video_frame")
        self.video_frame.setGeometry(QRect(10, 10, 381, 381))
        self.video_frame.setFrameShape(QFrame.Panel)
        self.video_frame.setFrameShadow(QFrame.Sunken)
        self.video_frame.setLineWidth(1)
        self.line_2 = QFrame(w_ImageCollector)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(400, 10, 16, 381))
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.plainTextEdit = QPlainTextEdit(w_ImageCollector)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(420, 260, 231, 131))
        self.plainTextEdit.setReadOnly(True)
        self.label = QLabel(w_ImageCollector)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(420, 235, 41, 16))
        self.active_coin_combo_box = QComboBox(w_ImageCollector)
        self.active_coin_combo_box.setObjectName(u"active_coin_combo_box")
        self.active_coin_combo_box.setGeometry(QRect(460, 230, 191, 22))
        self.tabWidget = QTabWidget(w_ImageCollector)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QRect(420, 10, 231, 211))
        self.camera_tab = QWidget()
        self.camera_tab.setObjectName(u"camera_tab")
        self.camera_swich_combo_box = QComboBox(self.camera_tab)
        self.camera_swich_combo_box.setObjectName(u"camera_swich_combo_box")
        self.camera_swich_combo_box.setGeometry(QRect(90, 10, 131, 21))
        self.label_2 = QLabel(self.camera_tab)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 10, 61, 20))
        self.label_2.setAlignment(Qt.AlignCenter)
        self.save_photo_button = QPushButton(self.camera_tab)
        self.save_photo_button.setObjectName(u"save_photo_button")
        self.save_photo_button.setEnabled(True)
        self.save_photo_button.setGeometry(QRect(10, 60, 211, 31))
        self.save_photo_button.setCheckable(False)
        self.auto_mark_edges_checkbox = QCheckBox(self.camera_tab)
        self.auto_mark_edges_checkbox.setObjectName(u"auto_mark_edges_checkbox")
        self.auto_mark_edges_checkbox.setEnabled(False)
        self.auto_mark_edges_checkbox.setGeometry(QRect(10, 30, 221, 31))
        self.auto_mark_edges_checkbox.setCheckable(False)
        self.new_coin_button = QPushButton(self.camera_tab)
        self.new_coin_button.setObjectName(u"new_coin_button")
        self.new_coin_button.setEnabled(True)
        self.new_coin_button.setGeometry(QRect(10, 100, 211, 31))
        self.color_correction_button = QPushButton(self.camera_tab)
        self.color_correction_button.setObjectName(u"color_correction_button")
        self.color_correction_button.setEnabled(False)
        self.color_correction_button.setGeometry(QRect(10, 140, 211, 31))
        self.tabWidget.addTab(self.camera_tab, "")
        self.gallery_tab = QWidget()
        self.gallery_tab.setObjectName(u"gallery_tab")
        self.mark_button = QPushButton(self.gallery_tab)
        self.mark_button.setObjectName(u"mark_button")
        self.mark_button.setGeometry(QRect(10, 50, 211, 31))
        self.next_gallery_photo_button = QPushButton(self.gallery_tab)
        self.next_gallery_photo_button.setObjectName(u"next_gallery_photo_button")
        self.next_gallery_photo_button.setGeometry(QRect(120, 10, 101, 31))
        self.previous_gallery_photo_button = QPushButton(self.gallery_tab)
        self.previous_gallery_photo_button.setObjectName(u"previous_gallery_photo_button")
        self.previous_gallery_photo_button.setGeometry(QRect(10, 10, 91, 31))
        self.tabWidget.addTab(self.gallery_tab, "")
        self.coin_tab = QWidget()
        self.coin_tab.setObjectName(u"coin_tab")
        self.tabWidget.addTab(self.coin_tab, "")

        self.retranslateUi(w_ImageCollector)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(w_ImageCollector)
    # setupUi

    def retranslateUi(self, w_ImageCollector):
        w_ImageCollector.setWindowTitle(QCoreApplication.translate("w_ImageCollector", u"Form", None))
        self.plainTextEdit.setPlainText("")
        self.label.setText(QCoreApplication.translate("w_ImageCollector", u"Coin", None))
        self.label_2.setText(QCoreApplication.translate("w_ImageCollector", u"Camera", None))
        self.save_photo_button.setText(QCoreApplication.translate("w_ImageCollector", u"New Photo", None))
        self.auto_mark_edges_checkbox.setText(QCoreApplication.translate("w_ImageCollector", u"auto-mark coin edges", None))
        self.new_coin_button.setText(QCoreApplication.translate("w_ImageCollector", u"New Coin", None))
        self.color_correction_button.setText(QCoreApplication.translate("w_ImageCollector", u"Automatic Color Correction", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.camera_tab), QCoreApplication.translate("w_ImageCollector", u"Camera", None))
        self.mark_button.setText(QCoreApplication.translate("w_ImageCollector", u"Mark Coin Edges", None))
        self.next_gallery_photo_button.setText(QCoreApplication.translate("w_ImageCollector", u"Next Photo", None))
        self.previous_gallery_photo_button.setText(QCoreApplication.translate("w_ImageCollector", u"Previous Photo", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.gallery_tab), QCoreApplication.translate("w_ImageCollector", u"Gallery", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.coin_tab), QCoreApplication.translate("w_ImageCollector", u"Coin ", None))
    # retranslateUi

