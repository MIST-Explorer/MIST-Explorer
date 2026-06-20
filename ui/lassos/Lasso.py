from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal
import colorsys
import math
import random


def pick_distinct_color(existing_rgb_tuples, num_candidates=20):
    """Return (r, g, b, 50) maximally distinct from existing_rgb_tuples.

    Generates candidates in HSL space with controlled saturation/lightness
    for vibrancy, then returns the one with maximum minimum Euclidean RGB
    distance to all existing colors.
    """
    best, best_dist = None, -1
    for _ in range(num_candidates):
        h = random.random()
        s = random.uniform(0.5, 1.0)
        lightness = random.uniform(0.35, 0.65)
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        candidate = (int(r * 255), int(g * 255), int(b * 255))
        if not existing_rgb_tuples:
            return candidate + (50,)
        min_dist = min(
            math.sqrt(sum((a - b_) ** 2 for a, b_ in zip(candidate, ec)))
            for ec in existing_rgb_tuples
        )
        if min_dist > best_dist:
            best_dist, best = min_dist, candidate
    return best + (50,)


class Lasso(QRubberBand):
    def __init__(self, shape: QRubberBand.Shape, parent=None, existing_colors=None):
        super().__init__(shape, parent)
        col = pick_distinct_color(existing_colors or [])[:3]
        self.color = QColor(*col)
        self.filled = False
        self.dragging_threshold = 5
        self.mouse_press_pos = None
        self.mouse_move_pos = None
        self.is_dragging = False

    roi_moved = pyqtSignal()

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
        if (
            self.mouse_press_pos
            and (event.pos() - self.mouse_press_pos).manhattanLength()
            > self.dragging_threshold
        ):
            self.roi_moved.emit()
            event.ignore()
        self.mouse_press_pos = None
        self.mouse_move_pos = None
