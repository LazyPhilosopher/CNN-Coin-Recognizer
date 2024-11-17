import logging
from multiprocessing import Process, Queue
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot, QProcess
from PySide6.QtWidgets import QApplication
from numpy import ndarray

# Mock imports for unavailable modules in the provided code
# These should be replaced with actual imports
from core.qt_communication.base import *
from core.qt_communication.messages.processing_module.Requests import RemoveBackgroundRequest
from core.qt_communication.messages.processing_module.Responses import ProcessedImageResponse
from core.utilities.helper import qimage_to_cv2, remove_background_rembg, cv2_to_qimage


# ANSI escape codes for coloring
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

# qt_signals = CommonSignals()

# Custom Log Formatter to color processName
class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.processName = f"{COLOR_GREEN}{record.processName}{COLOR_RESET}"
        return super().format(record)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(processName)s - PID:%(process)d] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter('%(asctime)s - [%(processName)s - PID:%(process)d] - %(message)s', datefmt="%H:%M:%S"))


class Worker:
    def __init__(self, func: Callable, arguments: dict, response_queue: Queue):
        self.func = func
        self.arguments = arguments
        self.response_queue = response_queue

    def run(self):
        worker_name = f"{COLOR_BLUE}Worker{COLOR_RESET}"
        logging.info(f"{worker_name}: {COLOR_BLUE}started.{COLOR_RESET}")
        response = self.func(**self.arguments)
        self.response_queue.put(response)  # Send the response to the queue
        logging.info(f"{worker_name}: {COLOR_BLUE}finished.{COLOR_RESET}")




class ResponseMonitor(QThread):
    response_received = Signal(object)

    def __init__(self, response_queue: Queue):
        super().__init__()
        self.response_queue = response_queue
        self.running = True

    def run(self):
        while self.running:
            if not self.response_queue.empty():
                response = self.response_queue.get()
                self.response_received.emit(response)

    def stop(self):
        self.running = False
        self.wait()


class ProcessingModule(QObject):
    def __init__(self):
        super().__init__()
        self.name = f"{COLOR_BLUE}ProcessingModule{COLOR_RESET}"
        self.is_running = False
        self.max_processes = 5
        self.response_queue = Queue()
        self.response_monitor = ResponseMonitor(self.response_queue)
        self.response_monitor.response_received.connect(self.handle_new_response)

        self.running_processes = []  # Track running QProcess instances
        self.pending_tasks = []  # Queue for pending tasks

    def start(self):
        self.is_running = True
        self.response_monitor.start()
        logging.info(f"{self.name}: Processing module started.")

    def stop(self):
        self.is_running = False
        for process in self.running_processes:
            process.terminate()
        self.running_processes.clear()
        self.response_monitor.stop()
        logging.info(f"{self.name}: Processing module stopped.")

    def _process_pending_tasks(self):
        if len(self.running_processes) < self.max_processes and self.pending_tasks:
            handler, request = self.pending_tasks.pop(0)

            cv2_image = qimage_to_cv2(request.picture)
            request_dict = {"image": cv2_image}

            worker = Worker(func=handler, arguments=request_dict, response_queue=self.response_queue)
            worker_process = Process(target=worker.run, name=f"{COLOR_GREEN}Worker_{COLOR_RESET}")
            worker_process.start()

            self.running_processes.append(worker_process)
            logging.info(f"{self.name}: {COLOR_BLUE}Started Worker process.{COLOR_RESET}")

    @Slot(object)
    def handle_new_response(self, response):
        logging.info(f"{self.name}: New response received: {response}")
        # Add your logic to process the response

    def _handle_remove_background_request(self, image: ndarray):
        logging.info(f"Handling RemoveBackgroundRequest")
        return remove_background_rembg(image)

