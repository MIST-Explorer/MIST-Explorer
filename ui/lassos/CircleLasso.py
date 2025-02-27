from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QPainter, QPen, QColor
import random

from ui.lassos.Lasso import Lasso

class CircleLasso(Lasso):
    def __init__(self, parent=None):
        super().__init__(QRubberBand.Shape.Rectangle, parent)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawEllipse(self.rect())

        if self.filled:
            color_trans = QColor(self.color)
            color_trans.setAlpha(75)
            painter.setBrush(color_trans)
            painter.drawEllipse(self.rect())


