# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_ImageCollector.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
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
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QStatusBar, QTabWidget, QWidget)

class Ui_ImageCollector(object):
    def setupUi(self, ImageCollector):
        if not ImageCollector.objectName():
            ImageCollector.setObjectName(u"ImageCollector")
        ImageCollector.resize(995, 508)
        self.centralwidget = QWidget(ImageCollector)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(750, 290, 49, 16))
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(730, 5, 16, 471))
        self.line_2.setFrameShape(QFrame.Shape.VLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QRect(750, 5, 231, 271))
        self.camera_tab = QWidget()
        self.camera_tab.setObjectName(u"camera_tab")
        self.camera_swich_combo_box = QComboBox(self.camera_tab)
        self.camera_swich_combo_box.setObjectName(u"camera_swich_combo_box")
        self.camera_swich_combo_box.setGeometry(QRect(90, 10, 131, 21))
        self.label_2 = QLabel(self.camera_tab)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 10, 61, 20))
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.save_photo_button = QPushButton(self.camera_tab)
        self.save_photo_button.setObjectName(u"save_photo_button")
        self.save_photo_button.setEnabled(True)
        self.save_photo_button.setGeometry(QRect(10, 50, 211, 31))
        self.save_photo_button.setCheckable(False)
        self.auto_background_deletion_checkbox = QCheckBox(self.camera_tab)
        self.auto_background_deletion_checkbox.setObjectName(u"auto_background_deletion_checkbox")
        self.auto_background_deletion_checkbox.setEnabled(True)
        self.auto_background_deletion_checkbox.setGeometry(QRect(10, 120, 201, 31))
        self.auto_background_deletion_checkbox.setCheckable(True)
        self.new_coin_button = QPushButton(self.camera_tab)
        self.new_coin_button.setObjectName(u"new_coin_button")
        self.new_coin_button.setEnabled(True)
        self.new_coin_button.setGeometry(QRect(10, 90, 211, 31))
        self.tabWidget.addTab(self.camera_tab, "")
        self.gallery_tab = QWidget()
        self.gallery_tab.setObjectName(u"gallery_tab")
        self.next_gallery_photo_button = QPushButton(self.gallery_tab)
        self.next_gallery_photo_button.setObjectName(u"next_gallery_photo_button")
        self.next_gallery_photo_button.setGeometry(QRect(120, 10, 101, 31))
        self.previous_gallery_photo_button = QPushButton(self.gallery_tab)
        self.previous_gallery_photo_button.setObjectName(u"previous_gallery_photo_button")
        self.previous_gallery_photo_button.setGeometry(QRect(10, 10, 91, 31))
        self.delete_background_button = QPushButton(self.gallery_tab)
        self.delete_background_button.setObjectName(u"delete_background_button")
        self.delete_background_button.setGeometry(QRect(10, 50, 211, 31))
        self.reset_button = QPushButton(self.gallery_tab)
        self.reset_button.setObjectName(u"reset_button")
        self.reset_button.setGeometry(QRect(10, 170, 211, 31))
        self.gallery_save_button = QPushButton(self.gallery_tab)
        self.gallery_save_button.setObjectName(u"gallery_save_button")
        self.gallery_save_button.setGeometry(QRect(10, 210, 211, 31))
        palette = QPalette()
        brush = QBrush(QColor(255, 0, 0, 50))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush)
        self.gallery_save_button.setPalette(palette)
        self.reset_vertices_button = QPushButton(self.gallery_tab)
        self.reset_vertices_button.setObjectName(u"reset_vertices_button")
        self.reset_vertices_button.setEnabled(False)
        self.reset_vertices_button.setGeometry(QRect(10, 90, 211, 31))
        self.crop_with_vertices_button = QPushButton(self.gallery_tab)
        self.crop_with_vertices_button.setObjectName(u"crop_with_vertices_button")
        self.crop_with_vertices_button.setEnabled(False)
        self.crop_with_vertices_button.setGeometry(QRect(10, 130, 211, 31))
        self.tabWidget.addTab(self.gallery_tab, "")
        self.coin_catalog_year_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_year_dropbox.setObjectName(u"coin_catalog_year_dropbox")
        self.coin_catalog_year_dropbox.setGeometry(QRect(820, 350, 161, 22))
        self.video_frame = QFrame(self.centralwidget)
        self.video_frame.setObjectName(u"video_frame")
        self.video_frame.setGeometry(QRect(10, 0, 720, 480))
        self.video_frame.setFrameShape(QFrame.Shape.Panel)
        self.video_frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.video_frame.setLineWidth(1)
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(750, 320, 61, 16))
        self.coin_catalog_country_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_country_dropbox.setObjectName(u"coin_catalog_country_dropbox")
        self.coin_catalog_country_dropbox.setGeometry(QRect(820, 290, 161, 22))
        self.coin_catalog_name_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_name_dropbox.setObjectName(u"coin_catalog_name_dropbox")
        self.coin_catalog_name_dropbox.setGeometry(QRect(820, 320, 161, 22))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(750, 350, 31, 16))
        ImageCollector.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(ImageCollector)
        self.statusbar.setObjectName(u"statusbar")
        ImageCollector.setStatusBar(self.statusbar)

        self.retranslateUi(ImageCollector)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(ImageCollector)
    # setupUi

    def retranslateUi(self, ImageCollector):
        ImageCollector.setWindowTitle(QCoreApplication.translate("ImageCollector", u"Image Collector", None))
        self.label_3.setText(QCoreApplication.translate("ImageCollector", u"Country", None))
        self.label_2.setText(QCoreApplication.translate("ImageCollector", u"Camera", None))
        self.save_photo_button.setText(QCoreApplication.translate("ImageCollector", u"New Photo", None))
        self.auto_background_deletion_checkbox.setText(QCoreApplication.translate("ImageCollector", u"Delete Coin Background", None))
        self.new_coin_button.setText(QCoreApplication.translate("ImageCollector", u"New Coin", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.camera_tab), QCoreApplication.translate("ImageCollector", u"Camera", None))
        self.next_gallery_photo_button.setText(QCoreApplication.translate("ImageCollector", u"Next Photo", None))
        self.previous_gallery_photo_button.setText(QCoreApplication.translate("ImageCollector", u"Previous Photo", None))
        self.delete_background_button.setText(QCoreApplication.translate("ImageCollector", u"Automatic Background Removal", None))
        self.reset_button.setText(QCoreApplication.translate("ImageCollector", u"Reset to Uncropped Version", None))
        self.gallery_save_button.setText(QCoreApplication.translate("ImageCollector", u"Save", None))
        self.reset_vertices_button.setText(QCoreApplication.translate("ImageCollector", u"Reset Manual Verices", None))
        self.crop_with_vertices_button.setText(QCoreApplication.translate("ImageCollector", u"Crop With Vertices", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.gallery_tab), QCoreApplication.translate("ImageCollector", u"Gallery", None))
        self.label_4.setText(QCoreApplication.translate("ImageCollector", u"Coin name", None))
        self.label.setText(QCoreApplication.translate("ImageCollector", u"Year", None))
    # retranslateUi

