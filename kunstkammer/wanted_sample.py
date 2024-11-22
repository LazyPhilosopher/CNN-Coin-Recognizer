import logging
import random
import sys
import time
from multiprocessing import Process
from typing import Callable, List

from PySide6.QtCore import QObject, QCoreApplication, Signal, QTimer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class Signals(QObject):
    """A class to define shared signals."""
    log_signal = Signal(str)
    worker_done = Signal(object)


class Worker:
    def __init__(self, method: Callable, arguments: dict):
        self.method = method
        self.arguments = arguments

    def run(self):
        worker_name = f"Worker "
        logging.info(f"{worker_name}: started.")
        self.method(**self.arguments)
        logging.info(f"{worker_name}: finished.")


class WorkerManager:
    def __init__(self, signals, process_count=5):
        super().__init__()

        self.name = f"WorkerManager"
        self.max_processes = process_count
        self.signals = signals
        self.running_processes: List[Process] = []
        self.pending_tasks: List[(str, int, int)] = []

        self.signals.worker_done.connect(self._check_running_processes())

    def run_workers(self, pending_tasks):
        self.pending_tasks.extend(pending_tasks)
        self._assign_tasks()

    def _assign_tasks(self):
        while len(self.running_processes) < self.max_processes and self.pending_tasks:
            operation, num_a, num_b = self.pending_tasks.pop(0)

            match(operation):
                case '+':
                    method = self.add
                case '-':
                    method = self.sub

            worker = Worker(method=method, arguments={"num_a": num_a, "num_b": num_b})

            worker_process = Process(target=worker.run, name=f"Worker_A{operation}B:")
            worker_process.start()

            self.running_processes.append(worker_process)
            logging.info(f"{self.name}: Started Worker_A{operation}B.")

    def _check_running_processes(self):
        for process in self.running_processes[:]:
            if not process.is_alive():
                process.join()
                self.running_processes.remove(process)
                logging.info(f"{self.name}: Worker process completed and cleaned up.")

        self._assign_tasks()

        if not self.pending_tasks and not self.running_processes:
            logging.info(f"{self.name}: All tasks completed. Exiting application.")
            QCoreApplication.quit()

    def add(self, num_a, num_b):
        time.sleep(1)
        logger.info(f"A+B={num_a+num_b}")

    def sub(self, num_a, num_b):
        time.sleep(1)
        logger.info(f"A-B={num_a-num_b}")

    def handle_add_result(self, result):
        out = f"Result of A+B={result}."
        logger.info(out)
        self.signals.log_signal.emit(out)

    def handle_sub_result(self, result):
        out = f"Result of A-B={result}."
        logger.info(out)
        self.signals.log_signal.emit(out)


class LoggerModule(QObject):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.signals.log_signal.connect(self.log_func)

    def log_func(self, msg):
        logger.info(msg)


if __name__ == "__main__":
    app = QCoreApplication(sys.argv)

    signals = Signals()
    logger_module = LoggerModule(signals=signals)
    manager = WorkerManager(signals=signals, process_count=5)

    length = 50
    op_list = random.choices(["+", "-"], k=length)
    a_list = random.choices(range(1, 50), k=length)
    b_list = random.choices(range(1, 50), k=length)

    arguments = [(op, a, b) for op, a, b in zip(op_list, a_list, b_list)]
    manager.run_workers(arguments)

    sys.exit(app.exec())
