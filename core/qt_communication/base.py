from enum import Enum
from typing import Type

from PySide6.QtCore import QTimer, QEventLoop, Signal

from core.qt_communication.messages.common_signals import CommonSignals

qt_signals = CommonSignals()


class Modules(Enum):
    MAIN = 1
    VIDEO_STREAM = 2
    CATALOG_HANDLER = 3
    GALLERY_WINDOW = 4
    DRAGGABLE_CROSS_OVERLAY = 5
    PROCESSING_MODULE = 6
    IMAGE_COLLECTOR_WINDOW = 7


class MessageBase:
    def __init__(self):
        self.source: Modules | None = None
        self.destination: Modules | None = None


def blocking_response_message_await(request_signal: Signal,
                                    request_message: MessageBase,
                                    response_signal: Signal,
                                    response_message_type: Type[MessageBase],
                                    timeout_ms: int = 1000):
    ret_val: MessageBase | None = None
    loop: QEventLoop = QEventLoop()

    def _message_type_check(message: MessageBase):
        nonlocal ret_val
        if isinstance(message, response_message_type):
            ret_val = message
            loop.quit()

    # Set up the timeout mechanism using QTimer
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    response_signal.connect(_message_type_check)
    request_signal.emit(request_message)

    # Start the timer and the event loop
    timer.start(timeout_ms)
    loop.exec_()

    # Clean up
    response_signal.disconnect(_message_type_check)
    if timer.isActive():
        timer.stop()  # Stop the timer if the response was received before timeout

    return ret_val
