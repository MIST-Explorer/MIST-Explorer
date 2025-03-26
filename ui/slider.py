from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import pyqtSignal, pyqtSlot

class DoubleSlider(QSlider):
    doubleValueChanged = pyqtSignal(float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.valueChanged.connect(self.notifyValueChanged)

    @pyqtSlot(int)
    def notifyValueChanged(self, value: int):
        double_value = float(value/10.0)
        self.doubleValueChanged.emit(double_value)