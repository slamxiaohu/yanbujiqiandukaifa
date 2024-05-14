import sys
import demo

from PyQt5.QtWidgets import QApplication, QWidget

if __name__ == '__main__':

    app = QApplication(sys.argv)

    mainWindow = QWidget()

    ui = demo.Ui_Form()

    ui.setupUi(mainWindow)

    mainWindow.show()

    sys.exit(app.exec_())
