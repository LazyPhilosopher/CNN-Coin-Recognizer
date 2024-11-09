# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_augmentation_window.ui'
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
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QStatusBar,
    QWidget)

class Ui_AugmentationWindow(object):
    def setupUi(self, AugmentationWindow):
        if not AugmentationWindow.objectName():
            AugmentationWindow.setObjectName(u"AugmentationWindow")
        AugmentationWindow.resize(474, 224)
        self.centralwidget = QWidget(AugmentationWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.scrollArea = QScrollArea(self.centralwidget)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(10, 10, 171, 181))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 169, 179))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalSlider = QSlider(self.centralwidget)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setGeometry(QRect(300, 10, 160, 22))
        self.horizontalSlider.setOrientation(Qt.Horizontal)
        self.horizontalSlider_2 = QSlider(self.centralwidget)
        self.horizontalSlider_2.setObjectName(u"horizontalSlider_2")
        self.horizontalSlider_2.setGeometry(QRect(300, 40, 160, 22))
        self.horizontalSlider_2.setOrientation(Qt.Horizontal)
        self.horizontalSlider_3 = QSlider(self.centralwidget)
        self.horizontalSlider_3.setObjectName(u"horizontalSlider_3")
        self.horizontalSlider_3.setGeometry(QRect(300, 70, 160, 22))
        self.horizontalSlider_3.setOrientation(Qt.Horizontal)
        self.horizontalSlider_4 = QSlider(self.centralwidget)
        self.horizontalSlider_4.setObjectName(u"horizontalSlider_4")
        self.horizontalSlider_4.setGeometry(QRect(300, 100, 160, 22))
        self.horizontalSlider_4.setOrientation(Qt.Horizontal)
        self.generate_augmented_data_button = QPushButton(self.centralwidget)
        self.generate_augmented_data_button.setObjectName(u"generate_augmented_data_button")
        self.generate_augmented_data_button.setGeometry(QRect(270, 160, 161, 31))
        self.horizontalSlider_5 = QSlider(self.centralwidget)
        self.horizontalSlider_5.setObjectName(u"horizontalSlider_5")
        self.horizontalSlider_5.setGeometry(QRect(300, 130, 160, 22))
        self.horizontalSlider_5.setOrientation(Qt.Horizontal)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(200, 10, 81, 16))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(200, 40, 81, 16))
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(200, 70, 81, 16))
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(200, 100, 81, 16))
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(200, 130, 81, 16))
        AugmentationWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(AugmentationWindow)
        self.statusbar.setObjectName(u"statusbar")
        AugmentationWindow.setStatusBar(self.statusbar)

        self.retranslateUi(AugmentationWindow)

        QMetaObject.connectSlotsByName(AugmentationWindow)
    # setupUi

    def retranslateUi(self, AugmentationWindow):
        AugmentationWindow.setWindowTitle(QCoreApplication.translate("AugmentationWindow", u"Generate Augmented Data", None))
        self.generate_augmented_data_button.setText(QCoreApplication.translate("AugmentationWindow", u"Generate", None))
        self.label.setText(QCoreApplication.translate("AugmentationWindow", u"Max. Rotation", None))
        self.label_2.setText(QCoreApplication.translate("AugmentationWindow", u"Max. Distorsion", None))
        self.label_3.setText(QCoreApplication.translate("AugmentationWindow", u"Max. Blur", None))
        self.label_4.setText(QCoreApplication.translate("AugmentationWindow", u"Max. Noise", None))
        self.label_5.setText(QCoreApplication.translate("AugmentationWindow", u"Nr. of Pictures", None))
    # retranslateUi

