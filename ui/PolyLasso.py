from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPointF

class PolyLasso:
    def __init__(self):
        self.points = []  # Will store widget coordinates
        self.color = QColor(0, 0, 255, 100)
        self.completed = False

    def add_point(self, point):
        self.points.append(point)
        print(f"Added point: {point}")

    def draw(self, painter):
        if len(self.points) > 1:
            painter.setBrush(self.color)
            painter.setPen(QPen(Qt.GlobalColor.blue, 2))
            painter.drawPolygon(self.points)
