from PyQt6.QtWidgets import  QHBoxLayout, QGroupBox, QPushButton, QWidget, QComboBox, QDoubleSpinBox
from PyQt6.QtCore import pyqtSignal, QCoreApplication

class GaussianBlur(QWidget):
    confirm = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__()
        self.gaussian_blur = QGroupBox(parent)
        self.gaussian_blur.setTitle("Gaussian Blur")
        self.gaussian_blur.setMaximumWidth(200)
        self.gaussian_blur.setMaximumHeight(80)
        self.hor_layout = QHBoxLayout(self.gaussian_blur)
        self.components_layout = QHBoxLayout()
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setMaximum(1)
        self.spin_box.setSingleStep(0.1)
        self.combo_box = QComboBox()
        self.combo_box.setMinimumWidth(80)
        self.confirm = QPushButton()
        self.confirm.setText("Ok")
        self.components_layout.addWidget(self.spin_box)
        self.components_layout.addWidget(self.combo_box)
        self.components_layout.addWidget(self.confirm)
        self.components_layout.setSpacing(0)
        self.hor_layout.addLayout(self.components_layout)

    def updateChannelSelector(self, channels:dict):
        if self.combo_box.count() == 0:
            self.combo_box.addItems(list(channels.keys()))

