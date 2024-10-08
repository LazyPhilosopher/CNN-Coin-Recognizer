# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_main_window.designer'
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
from PySide6.QtWidgets import (QApplication, QFrame, QMainWindow, QPushButton,
    QSizePolicy, QStatusBar, QTabWidget, QTextEdit,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(433, 347)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setTabShape(QTabWidget.Rounded)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.AddImageButton = QPushButton(self.centralwidget)
        self.AddImageButton.setObjectName(u"AddImageButton")
        self.AddImageButton.setGeometry(QRect(30, 20, 171, 41))
        self.AddImageButton.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.ImageGalleryButton = QPushButton(self.centralwidget)
        self.ImageGalleryButton.setObjectName(u"ImageGalleryButton")
        self.ImageGalleryButton.setGeometry(QRect(30, 70, 171, 41))
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(30, 120, 171, 41))
        self.pushButton_3 = QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(230, 20, 171, 41))
        self.pushButton_4 = QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(230, 70, 171, 41))
        self.pushButton_5 = QPushButton(self.centralwidget)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(230, 120, 171, 41))
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(195, 20, 41, 301))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(30, 170, 371, 16))
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(30, 190, 171, 131))
        self.textEdit_2 = QTextEdit(self.centralwidget)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(230, 190, 171, 131))
        MainWindow.setCentralWidget(self.centralwidget)
        self.AddImageButton.raise_()
        self.ImageGalleryButton.raise_()
        self.pushButton_2.raise_()
        self.pushButton_3.raise_()
        self.pushButton_4.raise_()
        self.pushButton_5.raise_()
        self.line_2.raise_()
        self.line.raise_()
        self.textEdit.raise_()
        self.textEdit_2.raise_()
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.AddImageButton.setText(QCoreApplication.translate("MainWindow", u"Add New Image", None))
        self.ImageGalleryButton.setText(QCoreApplication.translate("MainWindow", u"Image Gallery", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Coin Catalog", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Model Training", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Model Configuration", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Recognition Test", None))
    # retranslateUi

