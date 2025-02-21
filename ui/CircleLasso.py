from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import QPoint
import random

from ui.Lasso import Lasso

class CircleLasso(Lasso):
    def __init__(self, shape, parent=None):
        super().__init__(shape, parent)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawEllipse(self.rect())

