import shutil
import os
from PySide6.QtWidgets import QMessageBox, QApplication, QWidget

import win32com.client

def move_files(file_list, source_folder, destination_folder, create_dir=True):
    # Ensure destination folder exists if create_dir is True
    if create_dir and not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in file_list:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        if os.path.exists(source_path):
            # Move the file to the new destination
            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} -> {destination_path}")
        else:
            print(f"File not found: {source_path}")


def show_confirmation_dialog(parent: QWidget, title: str, message: str):
    # Create a message box
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Question)  # Use the question icon
    msg_box.setWindowTitle(title)
    msg_box.setText(message)

    # Add Yes and No buttons using QMessageBox.StandardButton
    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    msg_box.setDefaultButton(QMessageBox.StandardButton.No)  # Set No as the default button

    # Display the dialog and capture the user's response
    result = msg_box.exec()

    if result == QMessageBox.StandardButton.Yes:
        return True  # User clicked Yes
    else:
        return False  # User clicked No


# def on_usb_device_connected():
#     print("A USB device has been connected.")
#
#
# def monitor_usb_devices():
#     wmi = win32com.client.GetObject("winmgmts:")
#     watcher = wmi.ExecNotificationQuery(
#         "SELECT * FROM Win32_DeviceChangeEvent"
#     )
#
#     while True:
#         event = watcher.NextEvent()
#         if event:
#             on_usb_device_connected()
