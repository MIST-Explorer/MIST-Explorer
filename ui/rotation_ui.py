from PyQt6.QtWidgets import  QHBoxLayout, QGroupBox, QLineEdit, QWidget, QPushButton
from PyQt6.QtCore import QCoreApplication

class RotateUI(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.rotate_groupbox = QGroupBox(parent)
        self.rotate_groupbox.setTitle("Rotate")
        self.rotate_groupbox.setMaximumWidth(250)
        self.rotate_groupbox.setMaximumHeight(60)
        self.hor_layout = QHBoxLayout(self.rotate_groupbox)
        self.rotate_components_layout = QHBoxLayout()
        self.rotate_line_edit = QLineEdit()
        self.rotate_confirm = QPushButton()
        self.rotate_confirm.setText("Confirm")
        self.rotate_components_layout.addWidget(self.rotate_line_edit)
        self.rotate_components_layout.addWidget(self.rotate_confirm)
        self.hor_layout.addLayout(self.rotate_components_layout)

        self.__retranslateUI()

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.rotate_line_edit.setToolTip(_translate("MainWindow", "Enter degrees of rotation"))

