import logging
from multiprocessing import Process
from typing import List

from PySide6.QtCore import QObject, Signal
from sympy.physics.quantum.matrixutils import numpy_ndarray

from core.qt_communication.base import MessageBase, Modules
from core.qt_communication.messages.common_signals import CommonSignals
from core.qt_communication.messages.processing_module import Requests, Responses
from core.utilities.helper import qimage_to_cv2, remove_background_rembg, cv2_to_qimage

# ANSI escape codes for coloring
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

# Custom Log Formatter to color processName
class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.processName = f"{COLOR_GREEN}{record.processName}{COLOR_RESET}"
        return super().format(record)


# Configure logging to avoid overlapping output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(processName)s - PID:%(process)d] - %(message)s",
    datefmt="%H:%M:%S",
)


class Worker:
    def __init__(self, func: callable, args: dict, signal=None):
        self.func = func
        self.args = args
        self.qt_signals = CommonSignals()

    def run(self):
        worker_name = f"{COLOR_BLUE}Worker {COLOR_RESET}"
        logging.info(f"{worker_name}: {COLOR_BLUE}started.{COLOR_RESET}")
        self.func(self.args)
        logging.info(f"{worker_name}: {COLOR_BLUE}finished.{COLOR_RESET}")


class ProcessingModule(QObject):

    def __init__(self):
        super().__init__()

        self.name = f"{COLOR_BLUE}ProcessingModule{COLOR_RESET}"
        self.max_processes = 5
        self.running_processes = []
        self.pending_tasks: List[(callable, MessageBase)] = []

        self.qt_signals = CommonSignals()
        self.worker_finished = Signal(object)

        self.qt_signals.processing_module_request.connect(self._handle_request)
        self.worker_finished.connect(self._process_pending_tasks)


    def _handle_request(self, request: MessageBase):
        request_handlers = {
            Requests.RemoveBackgroundRequest: self._handle_remove_background,
            # Requests.AugmentCoinCatalogRequest: self._handle_catalog_augmentation_request
        }

        handler = request_handlers.get(type(request), None)

        if handler:
            self.pending_tasks.append((handler, request))

        self._process_pending_tasks()


    def _process_pending_tasks(self):
        if len(self.running_processes) < self.max_processes and self.pending_tasks:
            (handler, request) = self.pending_tasks.pop()
            handler(request)


    def _handle_remove_background(self, request: Requests.RemoveBackgroundRequest):
        image = qimage_to_cv2(request.picture)

        def func(qimage):
            cv2_image: numpy_ndarray = qimage_to_cv2(qimage)
            cv2_without_background = remove_background_rembg(cv2_image)
            image_without_background = cv2_to_qimage(cv2_without_background)
            self.qt_signals.processing_module_request.emit(
                Responses.ProcessedImageResponse(image=cv2_to_qimage(image_without_background),
                                                 source=Modules.PROCESSING_MODULE,
                                                 destination=request.source))
            self.worker_finished.emit()

        worker = Worker(func, image)

        # Start Worker in a new multiprocessing Process
        worker_process = Process(target=worker.run, name=f"{COLOR_GREEN}Worker_remove_background{COLOR_RESET}")
        worker_process.start()

        # Track the worker process
        self.running_processes.append(worker_process)
        logging.info(f"{self.name}: {COLOR_BLUE}Started Worker _handle_remove_background.{COLOR_RESET}")

        # self.qt_signals.processing_module_request.emit(
        #     Responses.ProcessedImageResponse(image=cv2_to_qimage(img_no_bg),
        #                            source=Modules.PROCESSING_MODULE,
        #                            destination=request.source))

    # def _handle_catalog_augmentation_request(self, request: Requests.AugmentCoinCatalogRequest):
    #     catalog_dict = parse_directory_into_dictionary(request.catalog_path)
    #     os.makedirs(os.path.join(request.catalog_path, "augmented"), exist_ok=True)
    #
    #     for country in catalog_dict.keys():
    #         for coin_name in catalog_dict[country].keys():
    #             for year in catalog_dict[country][coin_name].keys():
    #                 os.makedirs(os.path.join(request.catalog_path, country, coin_name, year), exist_ok=True)
    #
    #                 for coin_photo in catalog_dict[country][coin_name][year]["uncropped"]:
    #
    #                     if not coin_photo in catalog_dict[country][coin_name][year]["cropped"]:
    #                         continue
    #
    #                     cropped_coin_photo_path = os.path.join(request.catalog_path, country, coin_name, year,
    #                                                            "cropped", coin_photo)
    #                     uncropped_coin_photo_path = os.path.join(request.catalog_path, country, coin_name, year,
    #                                                              "uncropped", coin_photo)
    #
    #                     os.makedirs(os.path.join(request.catalog_path, "augmented", country, coin_name, year),
    #                                 exist_ok=True)
    #
    #                     cv2_uncropped_image: np.ndarray = qimage_to_cv2(QImage(uncropped_coin_photo_path))
    #                     cv2_cropped_image: np.ndarray = qimage_to_cv2(QImage(cropped_coin_photo_path))
    #
    #                     # Transformations
    #                     for i in range(10):
    #                         image = cv2_to_qimage(cv2_uncropped_image)
    #                         image.save(
    #                             os.path.join(request.catalog_path, "augmented", country, coin_name, year,
    #                                          f"{i}_full.png"))
    #
    #                         image = cv2_to_qimage(cv2_cropped_image)
    #                         image.save(
    #                             os.path.join(request.catalog_path, "augmented", country, coin_name, year,
    #                                          f"{i}_hue.png"))