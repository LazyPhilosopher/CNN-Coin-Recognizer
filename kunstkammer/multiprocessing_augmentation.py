import logging
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path
import tensorflow as tf

from PySide6.QtGui import QImage
from PySide6.QtCore import QTimer, QObject, QCoreApplication

from core.utilities.helper import parse_directory_into_dictionary, qimage_to_cv2, imgaug_transformation, cv2_to_qimage, \
    transparent_to_hue, transparent_to_mask

tf.config.set_visible_devices([], 'GPU')

# ANSI escape codes for coloring
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Color the processName for MainProcess (WorkerManager)
        # if record.processName == "MainProcess":
        #     record.processName = f"{COLOR_GREEN}{record.processName}{COLOR_RESET}"
        # else:
        #     # Color worker process names
        #     record.processName = f"{COLOR_GREEN}{record.processName}{COLOR_RESET}"

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
    normalized_path = os.path.normpath(path)
    components = normalized_path.split(os.sep)
    return components


def augment(augmentation_path, coin_dir, cropped_coin_photo_path, uncropped_coin_photo_path):
    uncropped_image = QImage(uncropped_coin_photo_path)
    cropped_image = QImage(cropped_coin_photo_path)
    filename_without_extension = os.path.splitext(os.path.basename(uncropped_coin_photo_path))[0]

    cv2_uncropped_image = qimage_to_cv2(uncropped_image)
    cv2_cropped_image = qimage_to_cv2(cropped_image)
    cv2_cropped_mask = transparent_to_mask(cv2_cropped_image)

    for i in range(10):
        cv2_augmented_image, cv2_augmented_mask, cv2_augmented_crop = (
            imgaug_transformation(image=cv2_uncropped_image, mask=cv2_cropped_mask, transparent=cv2_cropped_image))

        image = cv2_to_qimage(cv2_augmented_image)
        # cv2_hue_image = transparent_to_hue(cv2_augmented_croped_image)
        mask = cv2_to_qimage(cv2_augmented_mask)
        crop = cv2_to_qimage(cv2_augmented_crop)

        image.save(
            os.path.join(f"{os.path.join(augmentation_path, 'images', coin_dir, filename_without_extension)}_{i}.png"))
        mask.save(
            os.path.join(f"{os.path.join(augmentation_path, 'masks', coin_dir, filename_without_extension)}_{i}.png"))
        crop.save(
            os.path.join(f"{os.path.join(augmentation_path, 'crops', coin_dir, filename_without_extension)}_{i}.png"))


def get_augmentation_tasks():
    out = []
    catalog_dict = parse_directory_into_dictionary(catalog_path)
    os.makedirs(os.path.join(catalog_path, "augmented"), exist_ok=True)
    os.makedirs(os.path.join(catalog_path, "augmented", "images"), exist_ok=True)
    os.makedirs(os.path.join(catalog_path, "augmented", "masks"), exist_ok=True)
    os.makedirs(os.path.join(catalog_path, "augmented", "crops"), exist_ok=True)

    for country in catalog_dict.keys():
        for coin_name in catalog_dict[country].keys():
            for year in catalog_dict[country][coin_name].keys():
                os.makedirs(os.path.join(catalog_path, country, coin_name, year), exist_ok=True)

                cropped_filenames = [path.name for path in catalog_dict[country][coin_name][year]["cropped"]]

                for coin_photo in catalog_dict[country][coin_name][year]["uncropped"]:

                    if not coin_photo.name in cropped_filenames:
                        continue

                    dir_name = str((country, coin_name, year)).replace("'", "")
                    augmentation_path = os.path.join(catalog_path, "augmented")
                    # mask_augmentation_path = os.path.join(catalog_path, "augmented", "masks", dir_name)
                    # cropped_augmentation_path = os.path.join(catalog_path, "augmented", "cropped", dir_name)
                    os.makedirs(os.path.join(catalog_path, "augmented", "images", dir_name), exist_ok=True)
                    os.makedirs(os.path.join(catalog_path, "augmented", "masks", dir_name), exist_ok=True)
                    os.makedirs(os.path.join(catalog_path, "augmented", "crops", dir_name), exist_ok=True)

                    # os.makedirs(mask_augmentation_path, exist_ok=True)
                    # os.makedirs(cropped_augmentation_path, exist_ok=True)

                    # cropped_coin_photo_path = os.path.join(catalog_path, country, coin_name, year, "cropped", coin_photo.name)
                    uncropped_coin_photo_path = coin_photo
                    cropped_coin_photo_path = os.path.join(catalog_path, country, coin_name, year, "cropped", coin_photo.name)

                    # augment(augmentation_path, cropped_coin_photo_path, uncropped_coin_photo_path)
                    out.append((augmentation_path, dir_name, uncropped_coin_photo_path, cropped_coin_photo_path))
    return out



class Worker:
    def __init__(self, name, augmentation_path, coin_dir, cropped_coin_photo_path, uncropped_coin_photo_path):
        self.name = name
        self.augmentation_path = augmentation_path
        self.coin_dir = coin_dir
        self.cropped_coin_photo_path = cropped_coin_photo_path
        self.uncropped_coin_photo_path = uncropped_coin_photo_path
        _, self.country, self.coin, self.year, _, _ = split_path(uncropped_coin_photo_path)

    def run(self):
        worker_name = f"{COLOR_BLUE}Worker [{self.name}] sec"
        logging.info(f"{worker_name}: {COLOR_BLUE}started.{COLOR_RESET}")
        augment(self.augmentation_path, self.coin_dir, self.cropped_coin_photo_path, self.uncropped_coin_photo_path)
        logging.info(f"{worker_name}: {COLOR_BLUE}finished.{COLOR_RESET}")


class WorkerManager(QObject):
    def __init__(self, process_count):
        super().__init__()
        # Colored name for MainProcess (WorkerManager)
        self.name = f"{COLOR_BLUE}WorkerManager"
        self.max_processes = process_count
        self.running_processes = []  # Store currently running worker processes
        self.pending_tasks = []  # Store pending tasks as a queue

        # Timer to periodically check for completed processes
        self.timer = QTimer()
        self.timer.timeout.connect(self._check_running_processes)
        self.timer.start(10)  # Check every 100 ms

    def run_workers(self, aug_tasks):
        # Queue up all tasks initially
        self.pending_tasks.extend(aug_tasks)
        self._assign_tasks()

    def _assign_tasks(self):
        # Assign tasks if there are available slots
        while len(self.running_processes) < self.max_processes and self.pending_tasks:
            augmentation_path, coin_dir, uncropped_coin_photo_path, cropped_coin_photo_path = self.pending_tasks.pop(0)
            _, country, coin, year, _, pic = split_path(uncropped_coin_photo_path)
            name = f"{country}/{coin}/{year}/{pic}"

            worker = Worker(name, augmentation_path, coin_dir, cropped_coin_photo_path, uncropped_coin_photo_path)

            # Start Worker in a new multiprocessing Process
            worker_process = Process(target=worker.run, name=f"{COLOR_GREEN}Worker [{name}]{COLOR_RESET}")
            worker_process.start()

            # Track the worker process
            self.running_processes.append(worker_process)
            logging.info(f"{self.name}: {COLOR_BLUE}Started Worker process for [{name}].{COLOR_RESET}")

    def _check_running_processes(self):
        # Check for completed processes
        for process in self.running_processes[:]:  # Copy list to allow modification
            if not process.is_alive():  # Process has finished
                process.join()  # Clean up the process
                self.running_processes.remove(process)
                logging.info(f"{self.name}: {COLOR_BLUE}Worker process completed and cleaned up.{COLOR_RESET}")

        # Assign more tasks if there are available slots
        self._assign_tasks()

        # Exit application if all tasks and processes are complete
        if not self.pending_tasks and not self.running_processes:
            logging.info(f"{self.name}: {COLOR_BLUE}All tasks completed. Exiting application.{COLOR_RESET}")
            self.timer.stop()
            QCoreApplication.quit()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print(f"Available devices: {tf.config.list_physical_devices()}")

    # physical_devices = tf.config.list_physical_devices('GPU')
    # if physical_devices:
    #     print(f"Running on GPU: {physical_devices}")
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     if gpus:
    #         try:
    #             for gpu in gpus:
    #                 tf.config.experimental.set_memory_growth(gpu, True)
    #         except RuntimeError as e:
    #             print(e)
    # else:
    #     print("Running on CPU")

    app = QCoreApplication(sys.argv)

    manager = WorkerManager(process_count=80)
    aug_tasks = get_augmentation_tasks()

    manager.run_workers(aug_tasks)

    sys.exit(app.exec())

