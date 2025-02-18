import logging
import os
import sys
import time
import uuid
from multiprocessing import Process, Pipe
from pathlib import Path
from random import uniform

import tensorflow as tf
from PySide6.QtCore import QTimer, QObject, QCoreApplication
from PySide6.QtGui import QImage

from core.utilities.helper import qimage_to_cv2, transparent_to_mask, imgaug_transformation, \
    cv2_to_qimage, parse_directory_into_dictionary

tf.config.set_visible_devices([], 'GPU')

# ANSI escape codes for coloring
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
COLOR_RED   = "\033[31m"

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.processName = f"{COLOR_GREEN}{record.processName}{COLOR_RESET}"
        return super().format(record)

def apply_rgb_mask(image_tensor, mask_tensor):
    """
    Masks an RGB image with a binary RGB mask. Keeps original pixel values where the mask is white (1, 1, 1),
    and sets to black (0, 0, 0) where the mask is black (0, 0, 0).

    Args:
        image_tensor: TensorFlow tensor of shape (height, width, 3) representing the original RGB image.
        mask_tensor: TensorFlow tensor of shape (height, width, 3) representing the binary RGB mask
                     with values either (1, 1, 1) or (0, 0, 0).

    Returns:
        A TensorFlow tensor of the same shape as the input, masked by the binary mask.
    """
    # Ensure mask is binary (1s or 0s)
    mask_bool = tf.reduce_all(mask_tensor == 1, axis=-1, keepdims=True)  # Shape: (height, width, 1)

    # Use tf.where to apply the mask
    result = tf.where(mask_bool, image_tensor, tf.zeros_like(image_tensor))

    return result

# Configure logging to avoid overlapping output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(processName)s - PID:%(process)d] - %(message)s",
    datefmt="%H:%M:%S",
)

# Apply custom formatter
logger = logging.getLogger()
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter('%(asctime)s - [%(processName)s - PID:%(process)d] - %(message)s', datefmt="%H:%M:%S"))

catalog_path = Path("coin_catalog")

def split_path(path):
    normalized_path = os.path.normpath(str(path))
    components = normalized_path.split(os.sep)
    return components

def augment(augmentation_path, coin_dir, cropped_coin_photo_path, uncropped_coin_photo_path, augmentation_config):
    uncropped_image = QImage(uncropped_coin_photo_path)
    cropped_image = QImage(cropped_coin_photo_path)
    filename_without_extension = os.path.splitext(os.path.basename(uncropped_coin_photo_path))[0]

    cv2_uncropped_image = qimage_to_cv2(uncropped_image)
    cv2_cropped_image = qimage_to_cv2(cropped_image)
    cv2_cropped_mask = transparent_to_mask(cv2_cropped_image)

    # Use the configuration value AUGMENTATION_AMOUNT (read from the config file)
    for i in range(augmentation_config["AUGMENTATION_AMOUNT"]):
        cv2_augmented_image, cv2_augmented_mask, cv2_augmented_crop = (
            imgaug_transformation(image=cv2_uncropped_image,
                                  mask=cv2_cropped_mask,
                                  transparent=cv2_cropped_image,
                                  scale_range=augmentation_config["SCALE_RANGE"],
                                  rotation_range=augmentation_config["ROTATION_RANGE"],
                                  gaussian_noise_range=augmentation_config["GAUSSIAN_NOISE_RANGE"],
                                  salt_and_pepper_noise_range=augmentation_config["SALT_AND_PEPPER_NOSE_RANGE"],
                                  poisson_noise_range=augmentation_config["POISSON_NOISE_RANGE"]))

        image = cv2_to_qimage(cv2_augmented_image)
        mask = cv2_to_qimage(cv2_augmented_mask)
        crop = cv2_to_qimage(cv2_augmented_crop)

        image_save_path = os.path.join(augmentation_path, 'images', coin_dir, f"{filename_without_extension}_{i}.png")
        mask_save_path  = os.path.join(augmentation_path, 'masks', coin_dir, f"{filename_without_extension}_{i}.png")
        crop_save_path  = os.path.join(augmentation_path, 'crops', coin_dir, f"{filename_without_extension}_{i}.png")

        error_amnt = 0
        while error_amnt < 3:
            image = image.save(image_save_path)
            mask = mask.save(mask_save_path)
            crop = crop.save(crop_save_path)

            if image and mask and crop:
                break
            else:
                logging.error(f"{COLOR_RED}Error: Was not able to save all three images!{COLOR_RESET}")
                error_amnt += 1
                time.sleep(1000)

def get_augmentation_tasks(picture_augmentation_amount: int):
    out = []
    catalog_dict = parse_directory_into_dictionary(catalog_path)
    augmentation_path = os.path.join(catalog_path, f"augmented_{picture_augmentation_amount}")
    os.makedirs(augmentation_path, exist_ok=True)
    os.makedirs(os.path.join(augmentation_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(augmentation_path, "masks"), exist_ok=True)
    os.makedirs(os.path.join(augmentation_path, "crops"), exist_ok=True)

    for country in catalog_dict.keys():
        for coin_name in catalog_dict[country].keys():
            for year in catalog_dict[country][coin_name].keys():
                os.makedirs(os.path.join(catalog_path, country, coin_name, year), exist_ok=True)

                cropped_filenames = [path.name for path in catalog_dict[country][coin_name][year]["cropped"]]

                for coin_photo in catalog_dict[country][coin_name][year]["uncropped"]:
                    if not coin_photo.name in cropped_filenames:
                        continue

                    dir_name = str((country, coin_name, year)).replace("'", "")

                    os.makedirs(os.path.join(augmentation_path, "images", dir_name), exist_ok=True)
                    os.makedirs(os.path.join(augmentation_path, "masks", dir_name), exist_ok=True)
                    os.makedirs(os.path.join(augmentation_path, "crops", dir_name), exist_ok=True)

                    uncropped_coin_photo_path = coin_photo
                    cropped_coin_photo_path = os.path.join(catalog_path, country, coin_name, year, "cropped", coin_photo.name)

                    out.append((augmentation_path, dir_name, uncropped_coin_photo_path, cropped_coin_photo_path))
    return out

class Worker:
    def __init__(self,
                 name: str,
                 process_pipe: Pipe,
                 augmentation_path: Path,
                 coin_dir: Path,
                 cropped_coin_photo_path: Path,
                 uncropped_coin_photo_path: Path,
                 augmentation_config: dict):
        self.name: str = name
        self.uuid = uuid.uuid4()
        self.process_pipe = process_pipe
        self.augmentation_path = augmentation_path
        self.coin_dir = coin_dir
        self.cropped_coin_photo_path = cropped_coin_photo_path
        self.uncropped_coin_photo_path = uncropped_coin_photo_path
        self.augmentation_config = augmentation_config
        _, self.country, self.coin, self.year, _, _ = split_path(uncropped_coin_photo_path)

    def run(self):
        worker_name = f"{COLOR_BLUE}Worker [{self.name}]{COLOR_RESET}"
        try:
            logging.info(f"{worker_name}: {COLOR_BLUE}started.{COLOR_RESET}")
            augment(self.augmentation_path, self.coin_dir, self.cropped_coin_photo_path, self.uncropped_coin_photo_path, self.augmentation_config)
            logging.info(f"{worker_name}: {COLOR_BLUE}finished.{COLOR_RESET}")
        except Exception as e:
            logging.error(f"{worker_name}: {COLOR_RED}Error: {e}{COLOR_RESET}")
        self.process_pipe.send(self.uuid)


class WorkerManager(QObject):
    def __init__(self, process_count, augmentation_config):
        super().__init__()
        self.name = f"{COLOR_BLUE}WorkerManager{COLOR_RESET}"
        self.max_processes = process_count
        self.augmentation_config = augmentation_config
        self.running_processes = {}  # Store currently running worker processes
        self.pending_tasks = []      # Store pending tasks as a queue

        # Timer to periodically check for completed processes
        self.timer = QTimer()
        self.timer.timeout.connect(self._check_running_processes)
        self.timer.start(100)

    def run_workers(self, aug_tasks):
        self.pending_tasks.extend(aug_tasks)
        self._assign_tasks()

    def _assign_tasks(self):
        while len(self.running_processes) < self.max_processes and self.pending_tasks:
            pending_task = self.pending_tasks.pop(0)
            augmentation_path, coin_dir, uncropped_coin_photo_path, cropped_coin_photo_path = pending_task
            _, country, coin, year, _, pic = split_path(uncropped_coin_photo_path)
            name = f"{country}/{coin}/{year}/{pic}"

            parent_conn, child_conn = Pipe(duplex=False)
            worker = Worker(name=name,
                            process_pipe=child_conn,
                            augmentation_path=augmentation_path,
                            coin_dir=coin_dir,
                            cropped_coin_photo_path=cropped_coin_photo_path,
                            uncropped_coin_photo_path=uncropped_coin_photo_path,
                            augmentation_config=self.augmentation_config)

            worker_process = Process(target=worker.run, name=f"{COLOR_GREEN}Worker [{name}]{COLOR_RESET}")
            worker_process.start()

            # Track the worker process
            self.running_processes[worker.uuid] = {"process": worker_process, "parent_conn": parent_conn, "child_conn": child_conn, "task": pending_task}
            logging.info(f"{self.name}: {COLOR_BLUE}Started Worker process for [{name}].{COLOR_RESET}")

    def _check_running_processes(self):
        # Check for completed processes
        termination_uuid_list = []
        for worker_uuid in self.running_processes.keys():  # Copy list to allow modification
            process = self.running_processes[worker_uuid]["process"]
            parent_conn = self.running_processes[worker_uuid]["parent_conn"]
            child_conn = self.running_processes[worker_uuid]["child_conn"]
            task = self.running_processes[worker_uuid]["task"]

            if not process.is_alive():  # Process was silently closed
                logging.error(f"Worker[{worker_uuid}] was closed silently. Will reschedule his task: {task}")
                self.pending_tasks.append(task)
                termination_uuid_list.append(worker_uuid)
                continue

            if parent_conn.poll() and parent_conn.recv() == worker_uuid:
                termination_uuid_list.append(worker_uuid)

        for worker_uuid in termination_uuid_list:
            process = self.running_processes[worker_uuid]["process"]
            parent_conn = self.running_processes[worker_uuid]["parent_conn"]
            child_conn = self.running_processes[worker_uuid]["child_conn"]

            process.join()  # Clean up the process
            process.terminate()
            parent_conn.close()
            child_conn.close()

            self.running_processes.pop(worker_uuid)
            logging.info(f"{self.name}: {COLOR_BLUE}Worker process completed and cleaned up.{COLOR_RESET}")

        # Assign more tasks if there are available slots
        self._assign_tasks()

        # Exit application if all tasks and processes are complete
        if not self.pending_tasks and not self.running_processes:
            logging.info(f"{self.name}: {COLOR_BLUE}All tasks completed. Exiting application.{COLOR_RESET}")
            self.timer.stop()
            QCoreApplication.quit()

def confirm_exit():
    os.system('pause')
    sys.exit(1)

def load_config() -> dict:
    """
    Reads the augmentation_config.txt file (which must reside in the same directory as the program)
    and validates that all required configuration keys are present and that their values are of the correct format.
    If a key is missing or a value is invalid, an error is logged and the program exits.
    """
    config_file = Path(__file__).resolve().parent / "augmentation_config.txt"
    if not config_file.is_file():
        logging.error("Configuration file augmentation_config.txt not found in the current directory.")
        confirm_exit()

    config_data = {}
    with config_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logging.error(f"Invalid configuration line: {line}")
                confirm_exit()
            key, val = line.split("=", 1)
            config_data[key.strip()] = val.strip()

    # Define the expected configuration keys and their requirements.
    config_schema = {
        "MAX_CONCURRENT_PROCESSES": {"type": int, "default": 16},
        "AUGMENTATION_AMOUNT": {"type": int, "default": 20},
        "SCALE_RANGE": {"type": tuple, "length": 2, "default": (0.8, 1.2)},
        "ROTATION_RANGE": {"type": tuple, "length": 2, "default": (-45, 45)},
        "GAUSSIAN_NOISE_RANGE": {"type": tuple, "length": 2, "default": (0, 0.1 * 255)},
        "SALT_AND_PEPPER_NOSE_RANGE": {"type": tuple, "length": 2, "default": (0.01, 0.05)},
        "POISSON_NOISE_RANGE": {"type": tuple, "length": 2, "default": (0, 8)}
    }

    config = {}
    for key, schema in config_schema.items():
        if key not in config_data:
            logging.error(f"Missing configuration key: {key}")
            confirm_exit()
        try:
            # Evaluate the value with restricted globals.
            # This allows arithmetic expressions like "0.1 * 255" to be computed.
            value = eval(config_data[key], {"__builtins__": {}})
        except Exception as e:
            logging.error(f"Error evaluating configuration key '{key}': {e}")
            confirm_exit()
        if schema["type"] == int:
            if not isinstance(value, int):
                logging.error(f"Configuration key '{key}' must be an integer. Got {value} of type {type(value)}")
                confirm_exit()
        elif schema["type"] == tuple:
            if not isinstance(value, tuple) or len(value) != schema["length"]:
                logging.error(f"Configuration key '{key}' must be a tuple of length {schema['length']}. Got {value}")
                confirm_exit()
            # Check that each element is a number (int or float)
            for element in value:
                if not isinstance(element, (int, float)):
                    logging.error(f"Configuration key '{key}' must be a tuple of numbers. Got element {element} of type {type(element)}")
                    confirm_exit()
        config[key] = value
    return config

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(f"Available devices: {tf.config.list_physical_devices()}")


    # Read and validate the configuration.
    config = load_config()
    app = QCoreApplication(sys.argv)

    # Use the value from the config file for process count.
    manager = WorkerManager(process_count=config["MAX_CONCURRENT_PROCESSES"], augmentation_config=config)
    aug_tasks = get_augmentation_tasks(config["AUGMENTATION_AMOUNT"])

    manager.run_workers(aug_tasks)

    sys.exit(app.exec())
