from PySide6.QtWidgets import QDialog, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout


class NewCoinDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Input Dialog")

        # Create text input fields
        self.coin_year_field = QLineEdit(self)
        self.coin_year_field.setPlaceholderText("Coin year")

        self.coin_country_field = QLineEdit(self)
        self.coin_country_field.setPlaceholderText("Coin country")

        self.coin_name_field = QLineEdit(self)
        self.coin_name_field.setPlaceholderText("Coin name")

        # Create buttons
        self.confirm_button = QPushButton("Confirm", self)
        self.cancel_button = QPushButton("Cancel", self)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.coin_year_field)
        layout.addWidget(self.coin_country_field)
        layout.addWidget(self.coin_name_field)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect button signals
        self.confirm_button.clicked.connect(self.on_confirm)
        self.cancel_button.clicked.connect(self.on_cancel)

    def on_confirm(self):
        # Print the values of the text fields
        # print(f"Coin year: {self.coin_year_field.text()}")
        # print(f"Coin country: {self.coin_country_field.text()}")
        # print(f"Coin name: {self.coin_name_field.text()}")
        self.accept()

    def on_cancel(self):
        self.reject()
