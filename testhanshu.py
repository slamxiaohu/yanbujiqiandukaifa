import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Resizable Window Example")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.label_fixed_size = QLabel("This label has a fixed size.")
        self.label_fixed_size.setFixedWidth(200)
        self.layout.addWidget(self.label_fixed_size)

        self.label_resizable = QLabel("This label is resizable.")
        self.layout.addWidget(self.label_resizable)
        self.maximize_button = QPushButton("Maximize")
        self.maximize_button.clicked.connect(self.toggle_maximized)
        self.layout.addWidget(self.maximize_button)

        self.hide_button = QPushButton("Hide")
        self.hide_button.clicked.connect(self.hide)
        self.layout.addWidget(self.hide_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        self.layout.addWidget(self.exit_button)

    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
            self.maximize_button.setText("Maximize")
        else:
            self.showMaximized()
            self.maximize_button.setText("Restore")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


