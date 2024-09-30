from PySide6.QtCore import QObject, Signal


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
