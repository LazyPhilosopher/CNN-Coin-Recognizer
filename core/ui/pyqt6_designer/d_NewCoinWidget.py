# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'd_NewCoinWidget.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QWidget)

class Ui_NewCoinWidget(object):
    def setupUi(self, NewCoinWidget):
        if not NewCoinWidget.objectName():
            NewCoinWidget.setObjectName(u"NewCoinWidget")
        NewCoinWidget.resize(267, 205)
        self.coin_name_field = QLineEdit(NewCoinWidget)
        self.coin_name_field.setObjectName(u"coin_name_field")
        self.coin_name_field.setGeometry(QRect(110, 10, 151, 31))
        self.coin_year_field = QLineEdit(NewCoinWidget)
        self.coin_year_field.setObjectName(u"coin_year_field")
        self.coin_year_field.setGeometry(QRect(110, 50, 151, 22))
        self.coin_country_field = QLineEdit(NewCoinWidget)
        self.coin_country_field.setObjectName(u"coin_country_field")
        self.coin_country_field.setGeometry(QRect(110, 80, 151, 22))
        self.coin_weight_field = QLineEdit(NewCoinWidget)
        self.coin_weight_field.setObjectName(u"coin_weight_field")
        self.coin_weight_field.setGeometry(QRect(110, 110, 151, 22))
        self.coin_content_field = QLineEdit(NewCoinWidget)
        self.coin_content_field.setObjectName(u"coin_content_field")
        self.coin_content_field.setGeometry(QRect(110, 140, 151, 22))
        self.line = QFrame(NewCoinWidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(90, 10, 16, 151))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.label = QLabel(NewCoinWidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 10, 71, 21))
        self.label_2 = QLabel(NewCoinWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 50, 49, 16))
        self.label_3 = QLabel(NewCoinWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 80, 49, 16))
        self.label_4 = QLabel(NewCoinWidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(20, 110, 71, 16))
        self.label_5 = QLabel(NewCoinWidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(20, 140, 49, 16))
        self.ok_button = QPushButton(NewCoinWidget)
        self.ok_button.setObjectName(u"ok_button")
        self.ok_button.setGeometry(QRect(90, 170, 91, 31))

        self.retranslateUi(NewCoinWidget)

        QMetaObject.connectSlotsByName(NewCoinWidget)
    # setupUi

    def retranslateUi(self, NewCoinWidget):
        NewCoinWidget.setWindowTitle(QCoreApplication.translate("NewCoinWidget", u"Form", None))
        self.label.setText(QCoreApplication.translate("NewCoinWidget", u"Coin name", None))
        self.label_2.setText(QCoreApplication.translate("NewCoinWidget", u"Year", None))
        self.label_3.setText(QCoreApplication.translate("NewCoinWidget", u"Country", None))
        self.label_4.setText(QCoreApplication.translate("NewCoinWidget", u"Weight [gr]", None))
        self.label_5.setText(QCoreApplication.translate("NewCoinWidget", u"Content", None))
        self.ok_button.setText(QCoreApplication.translate("NewCoinWidget", u"Create Coin", None))
    # retranslateUi

