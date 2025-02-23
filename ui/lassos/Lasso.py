from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import QPoint
import random

class Lasso(QRubberBand):
    def __init__(self, shape: QRubberBand.Shape, parent=None):
        super().__init__(shape, parent)
        self.color = QColor(*self._get_random_color()[:3])
        self.filled = False
        self.dragging_threshold = 5
        self.mouse_press_pos = None
        self.mouse_move_pos = None
        self.is_dragging = False
    
    def _get_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 50)
    
    def mousePressEvent(self, event):
        self.mouse_press_pos = event.pos()
        self.mouse_move_pos = event.pos() - self.pos()
        self.is_dragging = True

    def mouseMoveEvent(self, event):
        if self.is_dragging and self.mouse_press_pos:
            moved = event.pos() - self.mouse_press_pos
            if moved.manhattanLength() > self.dragging_threshold:
                self.move(event.pos() - self.mouse_move_pos)
                self.mouse_move_pos = event.pos() - self.pos()

    def set_filled(self, fill):
        self.filled = fill
        self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        if self.mouse_press_pos and (event.pos() - self.mouse_press_pos).manhattanLength() > self.dragging_threshold:
            event.ignore()
        self.mouse_press_pos = None
        self.mouse_move_pos = None