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

    # Processing Module
    processing_module_request = Signal(object)
    processing_module_response = Signal(object)
    worker_finished = Signal()

    # Video Module
    video_module_request = Signal(object)
    frame_available = Signal(object)

    def __init__(self):
        super().__init__()
