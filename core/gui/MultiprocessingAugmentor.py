from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QSlider

from core.gui.pyqt6_designer.d_Augmentation_Window import Ui_AugmentationWindow
from core.qt_communication.messages.common_signals import CommonSignals

from superqt import QLabeledRangeSlider

from core.utilities.multiprocessing_augmentation import WorkerManager, get_augmentation_tasks


class MultiprocessingAugmentor(QMainWindow, Ui_AugmentationWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.qt_signals = CommonSignals()

        self.selected_directory: str | None = None
        self._init()

    def _init(self):
        self.picture_amount_slider.valueChanged.connect(lambda x: self.number_of_pictures_label.setText(str(x)))

        self.scale_slider = self.replace_qslider_with_qlabelrangeslider(self.scale_slider, [1, 20], 1)
        self.rotation_slider = self.replace_qslider_with_qlabelrangeslider(self.rotation_slider, [-180, 180], 5)
        self.gaussian_noise_slider = self.replace_qslider_with_qlabelrangeslider(self.gaussian_noise_slider, [0, 50], 0.5)
        self.salt_and_pepper_slider = self.replace_qslider_with_qlabelrangeslider(self.salt_and_pepper_slider, [1, 10], 1)
        self.poisson_noise_slider = self.replace_qslider_with_qlabelrangeslider(self.poisson_noise_slider, [0, 15], 1)

        self.scale_slider.setValue([1,20])
        self.rotation_slider.setValue([-45,45])
        self.gaussian_noise_slider.setValue([0,25])
        self.salt_and_pepper_slider.setValue([1,5])
        self.poisson_noise_slider.setValue([0,8])

        self.number_of_pictures_label.setText(str(self.picture_amount_slider.value()))
        # print(self.noise_slider.maximum())

        self.dataset_directory_button.clicked.connect(self._open_directory_dialog)
        self.train_checkbox.stateChanged.connect(self._handle_nn_name_checkbox)
        self.generate_augmented_data_button.clicked.connect(self._generate_augmented_data)

    def _open_directory_dialog(self):
        """
        Opens a directory selection dialog and handles the chosen directory.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if directory:
            print("Selected Directory:", directory)
            if self._validate_directory_structure(Path(directory)):
                self.selected_directory = directory
                self.generate_augmented_data_button.setEnabled(True)
                self.log_text_window.appendPlainText(f"Selected directory: {self.selected_directory}")
            else:
                self.generate_augmented_data_button.setEnabled(False)

    def _validate_directory_structure(self, directory: Path):
        return True

    def _handle_nn_name_checkbox(self):
        if self.train_checkbox.isChecked():
            self.nn_name_line.setEnabled(True)
            self.generate_augmented_data_button.setText("Augment Images + Train NN")
        else:
            self.nn_name_line.setEnabled(False)
            self.generate_augmented_data_button.setText("Augment Images")

    def _generate_augmented_data(self):
        self.generate_augmented_data_button.setEnabled(False)
        self.log_text_window.appendPlainText("Start generating augmented data")
        print(f"Scale: {self.scale_slider.value()}")
        print(f"Rotation: {self.rotation_slider.value()}")
        print(f"Gaussian noise: {self.gaussian_noise_slider.value()}")
        print(f"Salt and Pepper noise: {self.salt_and_pepper_slider.value()}")
        print(f"Poisson noise: {self.poisson_noise_slider.value()}")

        augmentation_config = {
            "augmentation_amount": self.picture_amount_slider.value(),
            "scale_range": tuple(x / 10 for x in self.scale_slider.value()),
            "rotation_range": self.rotation_slider.value(),
            "gaussian_noise_range": self.gaussian_noise_slider.value(),
            "salt_and_pepper_noise_range": tuple(x / 100 for x in self.salt_and_pepper_slider.value()),
            "poisson_noise_range": self.poisson_noise_slider.value(),
        }

        manager = WorkerManager(process_count=32)
        manager.augmentation_config = augmentation_config

        aug_tasks = get_augmentation_tasks()
        manager.run_workers(aug_tasks)


    def replace_qslider_with_qlabelrangeslider(self, slider: QSlider, value_range: list[int], step: int | float) -> QLabeledRangeSlider:
        slider_parent = slider.parent()  # Get the parent widget.
        old_slider_geometry = slider.geometry()

        layout = slider_parent.layout()
        if layout is not None:
            # Find the index of the old slider in the layout.
            index = layout.indexOf(slider)
            # Remove the old slider from the layout.
            layout.removeWidget(slider)
        else:
            index = None

        # Hide and mark the old slider for deletion.
        slider.hide()
        slider.deleteLater()

        # Create the new QLabeledRangeSlider.
        new_slider = QLabeledRangeSlider(slider_parent)
        new_slider.setGeometry(old_slider_geometry)
        new_slider.setOrientation(Qt.Orientation.Horizontal)
        new_slider.setRange(*value_range)  # Set the initial range.
        new_slider.setPageStep(step)  # Set the initial range.

        # If the parent has a layout, insert the new slider at the old index.
        if layout is not None and index is not None:
            layout.insertWidget(index, new_slider)

        return new_slider
