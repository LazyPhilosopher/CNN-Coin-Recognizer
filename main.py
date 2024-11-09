import faulthandler

from core.app import OpenCV2CoinRecognizerApp

if __name__ == "__main__":
    faulthandler.enable()
    operator = OpenCV2CoinRecognizerApp()
