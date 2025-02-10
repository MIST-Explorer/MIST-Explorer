from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QPainter, QPen, QColor
import random

from ui.Lasso import Lasso


class RectLasso(Lasso):
    def __init__(self, shape, parent=None):
        super().__init__(shape, parent)
        
    def get_rubber_band_size_relative_to_scene(self):
        rect = self.geometry()
        top_left_scene = self.mapToScene(rect.topLeft())
        bottom_right_scene = self.mapToScene(rect.bottomRight())
        return bottom_right_scene.x() - top_left_scene.x(), bottom_right_scene.y() - top_left_scene.y()

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

    

    

    