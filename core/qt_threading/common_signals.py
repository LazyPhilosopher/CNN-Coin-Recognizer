from typing import Type

from PySide6.QtCore import QObject, Signal, QEventLoop

from core.qt_threading.headers.RequestBase import RequestBase


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class CommonSignals(QObject):

    # Available cameras:
    request_cameras_ids = Signal()
    response_camera_ids = Signal(object)

    # Catalog handler
    catalog_handler_request = Signal(object)
    catalog_handler_response = Signal(object)

    # Processing Module
    processing_module_request = Signal(object)

    # Video thread
    video_thread_request = Signal(object)
    video_thread_response = Signal(object)
    frame_available = Signal(object)

    def __init__(self):
        super().__init__()


def blocking_response_message_await(request_signal: Signal,
                                    request_message: RequestBase,
                                    response_signal: Signal,
                                    response_message_type: Type[RequestBase]):
    ret_val: RequestBase | None = None
    loop: QEventLoop = QEventLoop()

    def _message_type_check(message: RequestBase):
        nonlocal ret_val
        if isinstance(message, response_message_type):
            ret_val = message
            loop.quit()

    callback = lambda data: _message_type_check(data)
    response_signal.connect(callback)

    request_signal.emit(request_message)

    loop.exec_()
    response_signal.disconnect(callback)
    return ret_val
