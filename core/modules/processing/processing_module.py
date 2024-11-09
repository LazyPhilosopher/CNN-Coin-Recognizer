from PySide6.QtCore import QObject, QThread
from PySide6.QtWidgets import QApplication

from core.qt_communication.base import *
from core.qt_communication.messages.processing_module.Requests import *
from core.qt_communication.messages.processing_module.Responses import *
from core.utilities.helper import qimage_to_cv2, remove_background_rembg, cv2_to_qimage


class ProcessingModule(QObject):

    def __init__(self):
        super().__init__()

        self.is_running = False
        self.is_processing = False
        self.main_thread = QThread()
        self.qt_signals = CommonSignals()
        self.qt_signals.processing_module_request.connect(self.handle_request)

    def start_process(self):
        self.moveToThread(self.main_thread)
        self.main_thread.started.connect(self.worker)
        self.main_thread.start()
        self.is_running = True

    def worker(self):
        while self.is_running:
            QApplication.processEvents()

    def handle_request(self, request: MessageBase):
        request_handlers = {
            # GrayscalePictureRequest: self.handle_grayscale_picture,
            # DoNothingRequest: self.handle_do_nothing,
            RemoveBackgroundRequest: self._handle_remove_background,
            # RemoveBackgroundVerticesRequest: self.handle_remove_background_vertices,
            # AugmentedImageListRequest: self.handle_augmented_image_list_request
        }

        handler = request_handlers.get(type(request), None)
        if handler:
            if self.is_processing:
                return

            self.is_processing = True
            handler(request)
            self.is_processing = False

    def _handle_remove_background(self, request: RemoveBackgroundRequest):
        image = qimage_to_cv2(request.picture)
        img_no_bg = remove_background_rembg(image)
        self.qt_signals.processing_module_request.emit(
            ProcessedImageResponse(image=cv2_to_qimage(img_no_bg),
                                   source=Modules.PROCESSING_MODULE,
                                   destination=request.source))
