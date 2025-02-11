from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QPainter, QPen, QColor
import random

from ui.Lasso import Lasso


class RectLasso(Lasso):
    def __init__(self, parent=None):
        super().__init__(QRubberBand.Shape.Rectangle, parent)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.color)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(self.rect())

        if self.filled:
            color_trans = QColor(self.color)
            color_trans.setAlpha(75)
            painter.fillRect(self.rect(), color_trans)

    

    

    