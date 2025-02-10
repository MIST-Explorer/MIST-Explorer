from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QPainter, QPen, QColor
import random


class RectLasso(QRubberBand):
    def __init__(self, shape, parent=None):
        super().__init__(shape, parent)
        self.color = QColor(*self._get_random_color()[:3])
        self.filled = False
        self.dragging_threshold = 5
        self.mouse_press_pos = None
        self.mouse_move_pos = None
        self.is_dragging = False

    def _get_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 50)

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

    def set_filled(self, fill):
        self.filled = fill
        self.update()

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

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        if self.mouse_press_pos and (event.pos() - self.mouse_press_pos).manhattanLength() > self.dragging_threshold:
            event.ignore()
        self.mouse_press_pos = None
        self.mouse_move_pos = None