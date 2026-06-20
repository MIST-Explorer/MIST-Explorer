from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.slider import DoubleSlider


class GaussianBlur(QWidget):
    def __init__(self, parent, containing_layout: QVBoxLayout):
        super().__init__()
        self.gaussian_blur = QGroupBox(parent)
        self.gaussian_blur.setTitle("Smoothing")
        self.__layout = QHBoxLayout(self.gaussian_blur)
        self.components_layout = QVBoxLayout()
        self.combo_box = QComboBox()

        self.slider = DoubleSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(self.slider.TickPosition.TicksAbove)
        self.slider.setSingleStep(1)

        self.slider_label = QLabel("0 %")

        self.slider_layout = QHBoxLayout()
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addWidget(self.slider_label)

        self.confirm = QPushButton()
        self.confirm.setText("Replace with smoothed layer")

        self.components_layout.addLayout(self.slider_layout)
        self.components_layout.addWidget(self.confirm)
        self.components_layout.setSpacing(10)
        self.__layout.addLayout(self.components_layout)
        containing_layout.addWidget(self.gaussian_blur)

    def updateChannelSelector(self, channels: dict):
        self.combo_box.clear()
        channel_keys = sorted(
            channels.keys(), key=lambda x: int(x.replace("Channel ", ""))
        )
        self.combo_box.addItems(channel_keys)

    @pyqtSlot(float)
    def update_slider_label(self, value: float):
        self.slider_label.setText(str(value) + " %")
