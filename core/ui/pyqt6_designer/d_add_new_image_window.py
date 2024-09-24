# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_add_new_image_window.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QMainWindow, QSizePolicy,
    QStatusBar, QWidget)

class Ui_AddNewImageWindow(object):
    def setupUi(self, AddNewImageWindow):
        if not AddNewImageWindow.objectName():
            AddNewImageWindow.setObjectName(u"AddNewImageWindow")
        AddNewImageWindow.resize(452, 317)
        AddNewImageWindow.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(AddNewImageWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.video_frame = QFrame(self.centralwidget)
        self.video_frame.setObjectName(u"video_frame")
        self.video_frame.setGeometry(QRect(10, 10, 271, 271))
        self.video_frame.setFrameShape(QFrame.StyledPanel)
        self.video_frame.setFrameShadow(QFrame.Plain)
        AddNewImageWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(AddNewImageWindow)
        self.statusbar.setObjectName(u"statusbar")
        AddNewImageWindow.setStatusBar(self.statusbar)

        self.retranslateUi(AddNewImageWindow)

        QMetaObject.connectSlotsByName(AddNewImageWindow)
    # setupUi

    def retranslateUi(self, AddNewImageWindow):
        AddNewImageWindow.setWindowTitle(QCoreApplication.translate("AddNewImageWindow", u"MainWindow", None))
    # retranslateUi

