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
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QSlider, QStatusBar,
    QWidget)

class Ui_AugmentationWindow(object):
    def setupUi(self, AugmentationWindow):
        if not AugmentationWindow.objectName():
            AugmentationWindow.setObjectName(u"AugmentationWindow")
        AugmentationWindow.resize(739, 241)
        self.centralwidget = QWidget(AugmentationWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.rotation_slider = QSlider(self.centralwidget)
        self.rotation_slider.setObjectName(u"rotation_slider")
        self.rotation_slider.setGeometry(QRect(530, 10, 160, 22))
        self.rotation_slider.setOrientation(Qt.Horizontal)
        self.distortion_slider = QSlider(self.centralwidget)
        self.distortion_slider.setObjectName(u"distortion_slider")
        self.distortion_slider.setGeometry(QRect(530, 40, 160, 22))
        self.distortion_slider.setOrientation(Qt.Horizontal)
        self.blur_slider = QSlider(self.centralwidget)
        self.blur_slider.setObjectName(u"blur_slider")
        self.blur_slider.setGeometry(QRect(530, 70, 160, 22))
        self.blur_slider.setOrientation(Qt.Horizontal)
        self.noise_slider = QSlider(self.centralwidget)
        self.noise_slider.setObjectName(u"noise_slider")
        self.noise_slider.setGeometry(QRect(530, 100, 160, 22))
        self.noise_slider.setOrientation(Qt.Horizontal)
        self.generate_augmented_data_button = QPushButton(self.centralwidget)
        self.generate_augmented_data_button.setObjectName(u"generate_augmented_data_button")
        self.generate_augmented_data_button.setGeometry(QRect(500, 160, 161, 31))
        self.picture_amount_slider = QSlider(self.centralwidget)
        self.picture_amount_slider.setObjectName(u"picture_amount_slider")
        self.picture_amount_slider.setGeometry(QRect(530, 130, 160, 22))
        self.picture_amount_slider.setOrientation(Qt.Horizontal)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(430, 10, 81, 16))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(430, 40, 81, 16))
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(430, 70, 81, 16))
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(430, 100, 81, 16))
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(430, 130, 81, 16))
        self.rotation_label = QLabel(self.centralwidget)
        self.rotation_label.setObjectName(u"rotation_label")
        self.rotation_label.setGeometry(QRect(698, 10, 41, 20))
        self.distorsion_label = QLabel(self.centralwidget)
        self.distorsion_label.setObjectName(u"distorsion_label")
        self.distorsion_label.setGeometry(QRect(700, 40, 41, 16))
        self.blur_label = QLabel(self.centralwidget)
        self.blur_label.setObjectName(u"blur_label")
        self.blur_label.setGeometry(QRect(700, 70, 41, 16))
        self.noise_label = QLabel(self.centralwidget)
        self.noise_label.setObjectName(u"noise_label")
        self.noise_label.setGeometry(QRect(700, 100, 41, 16))
        self.number_of_pictures_label = QLabel(self.centralwidget)
        self.number_of_pictures_label.setObjectName(u"number_of_pictures_label")
        self.number_of_pictures_label.setGeometry(QRect(700, 130, 41, 16))
        self.augmented_color_frame = QFrame(self.centralwidget)
        self.augmented_color_frame.setObjectName(u"augmented_color_frame")
        self.augmented_color_frame.setGeometry(QRect(10, 10, 191, 201))
        self.augmented_color_frame.setFrameShape(QFrame.StyledPanel)
        self.augmented_color_frame.setFrameShadow(QFrame.Raised)
        self.augmented_hue_frame = QFrame(self.centralwidget)
        self.augmented_hue_frame.setObjectName(u"augmented_hue_frame")
        self.augmented_hue_frame.setGeometry(QRect(210, 10, 211, 201))
        self.augmented_hue_frame.setFrameShape(QFrame.StyledPanel)
        self.augmented_hue_frame.setFrameShadow(QFrame.Raised)
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
        self.rotation_label.setText(QCoreApplication.translate("AugmentationWindow", u"0", None))
        self.distorsion_label.setText(QCoreApplication.translate("AugmentationWindow", u"0", None))
        self.blur_label.setText(QCoreApplication.translate("AugmentationWindow", u"0", None))
        self.noise_label.setText(QCoreApplication.translate("AugmentationWindow", u"0", None))
        self.number_of_pictures_label.setText(QCoreApplication.translate("AugmentationWindow", u"0", None))
    # retranslateUi

