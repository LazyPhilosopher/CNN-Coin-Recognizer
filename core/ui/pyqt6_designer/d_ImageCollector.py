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
    QLabel, QMainWindow, QPlainTextEdit, QPushButton,
    QSizePolicy, QStatusBar, QTabWidget, QWidget)

class Ui_ImageCollector(object):
    def setupUi(self, ImageCollector):
        if not ImageCollector.objectName():
            ImageCollector.setObjectName(u"ImageCollector")
        ImageCollector.resize(664, 409)
        self.centralwidget = QWidget(ImageCollector)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(420, 255, 49, 16))
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(400, 5, 16, 381))
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QRect(420, 5, 231, 211))
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
        self.vertices_reset_button = QPushButton(self.gallery_tab)
        self.vertices_reset_button.setObjectName(u"vertices_reset_button")
        self.vertices_reset_button.setGeometry(QRect(10, 50, 211, 31))
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
        self.coin_catalog_year_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_year_dropbox.setObjectName(u"coin_catalog_year_dropbox")
        self.coin_catalog_year_dropbox.setGeometry(QRect(490, 225, 161, 22))
        self.video_frame = QFrame(self.centralwidget)
        self.video_frame.setObjectName(u"video_frame")
        self.video_frame.setGeometry(QRect(10, 5, 381, 381))
        self.video_frame.setFrameShape(QFrame.Panel)
        self.video_frame.setFrameShadow(QFrame.Sunken)
        self.video_frame.setLineWidth(1)
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(420, 285, 61, 16))
        self.plainTextEdit = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(420, 315, 231, 71))
        self.plainTextEdit.setReadOnly(True)
        self.coin_catalog_country_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_country_dropbox.setObjectName(u"coin_catalog_country_dropbox")
        self.coin_catalog_country_dropbox.setGeometry(QRect(490, 255, 161, 22))
        self.coin_catalog_name_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_name_dropbox.setObjectName(u"coin_catalog_name_dropbox")
        self.coin_catalog_name_dropbox.setGeometry(QRect(490, 285, 161, 22))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(420, 230, 31, 16))
        ImageCollector.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(ImageCollector)
        self.statusbar.setObjectName(u"statusbar")
        ImageCollector.setStatusBar(self.statusbar)

        self.retranslateUi(ImageCollector)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(ImageCollector)
    # setupUi

    def retranslateUi(self, ImageCollector):
        ImageCollector.setWindowTitle(QCoreApplication.translate("ImageCollector", u"MainWindow", None))
        self.label_3.setText(QCoreApplication.translate("ImageCollector", u"Country", None))
        self.label_2.setText(QCoreApplication.translate("ImageCollector", u"Camera", None))
        self.save_photo_button.setText(QCoreApplication.translate("ImageCollector", u"New Photo", None))
        self.auto_mark_edges_checkbox.setText(QCoreApplication.translate("ImageCollector", u"auto-mark coin edges", None))
        self.new_coin_button.setText(QCoreApplication.translate("ImageCollector", u"New Coin", None))
        self.color_correction_button.setText(QCoreApplication.translate("ImageCollector", u"Automatic Color Correction", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.camera_tab), QCoreApplication.translate("ImageCollector", u"Camera", None))
        self.vertices_reset_button.setText(QCoreApplication.translate("ImageCollector", u"Reset Coin Vertices", None))
        self.next_gallery_photo_button.setText(QCoreApplication.translate("ImageCollector", u"Next Photo", None))
        self.previous_gallery_photo_button.setText(QCoreApplication.translate("ImageCollector", u"Previous Photo", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.gallery_tab), QCoreApplication.translate("ImageCollector", u"Gallery", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.coin_tab), QCoreApplication.translate("ImageCollector", u"Coin ", None))
        self.label_4.setText(QCoreApplication.translate("ImageCollector", u"Coin name", None))
        self.plainTextEdit.setPlainText("")
        self.label.setText(QCoreApplication.translate("ImageCollector", u"Year", None))
    # retranslateUi

