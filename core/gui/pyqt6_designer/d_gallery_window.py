# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_gallery_window.designer'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_GalleryWindow(object):
    def setupUi(self, GalleryWindow):
        if not GalleryWindow.objectName():
            GalleryWindow.setObjectName(u"GalleryWindow")
        GalleryWindow.resize(512, 353)
        self.centralwidget = QWidget(GalleryWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.next_button = QPushButton(self.centralwidget)
        self.next_button.setObjectName(u"next_button")
        self.next_button.setGeometry(QRect(430, 260, 71, 71))
        self.previous_button = QPushButton(self.centralwidget)
        self.previous_button.setObjectName(u"previous_button")
        self.previous_button.setGeometry(QRect(340, 260, 71, 71))
        self.coin_catalog_year_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_year_dropbox.setObjectName(u"coin_catalog_year_dropbox")
        self.coin_catalog_year_dropbox.setGeometry(QRect(340, 30, 161, 31))
        self.image_frame = QFrame(self.centralwidget)
        self.image_frame.setObjectName(u"image_frame")
        self.image_frame.setGeometry(QRect(10, 10, 321, 321))
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Raised)
        self.coin_catalog_country_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_country_dropbox.setObjectName(u"coin_catalog_country_dropbox")
        self.coin_catalog_country_dropbox.setGeometry(QRect(340, 80, 161, 31))
        self.coin_catalog_name_dropbox = QComboBox(self.centralwidget)
        self.coin_catalog_name_dropbox.setObjectName(u"coin_catalog_name_dropbox")
        self.coin_catalog_name_dropbox.setGeometry(QRect(340, 130, 161, 31))
        self.reset_coin_vertices_button = QPushButton(self.centralwidget)
        self.reset_coin_vertices_button.setObjectName(u"reset_coin_vertices_button")
        self.reset_coin_vertices_button.setGeometry(QRect(340, 170, 161, 31))
        GalleryWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(GalleryWindow)
        self.statusbar.setObjectName(u"statusbar")
        GalleryWindow.setStatusBar(self.statusbar)

        self.retranslateUi(GalleryWindow)

        QMetaObject.connectSlotsByName(GalleryWindow)
    # setupUi

    def retranslateUi(self, GalleryWindow):
        GalleryWindow.setWindowTitle(QCoreApplication.translate("GalleryWindow", u"Gallery Window", None))
        self.next_button.setText(QCoreApplication.translate("GalleryWindow", u"Next", None))
        self.previous_button.setText(QCoreApplication.translate("GalleryWindow", u"Previous", None))
        self.reset_coin_vertices_button.setText(QCoreApplication.translate("GalleryWindow", u"Erase Vertices", None))
    # retranslateUi

