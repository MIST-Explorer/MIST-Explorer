from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtGui import QPainter, QPolygonF, QColor, QPen
from PyQt6.QtCore import Qt, QPointF



class PolyLasso(QWidget):
    def __init__(self):
        super().__init__()
        self.points = []
        self.selected_point = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_point = self.find_point(event.position())
            if clicked_point is not None:
                self.selected_point = clicked_point
            else:
                self.points.append(QPointF(event.position()))
            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            clicked_point = self.find_point(event.position())
            if clicked_point is not None:
                self.points.remove(clicked_point)
                self.selected_point = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.selected_point is not None:
            self.selected_point.setX(event.position().x())
            self.selected_point.setY(event.position().y())
            self.update()

    def mouseReleaseEvent(self, event):
        self.selected_point = None

    def find_point(self, pos):
        for point in self.points:
            if (point - pos).manhattanLength() < 10:
                return point
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw shaded polygon if there are at least two points
        if len(self.points) > 1:
            painter.setBrush(QColor(0, 0, 255, 100))  # Semi-transparent blue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygonF(self.points))

        # Draw points
        for point in self.points:
            painter.setBrush(QColor(255, 255, 255) if point == self.selected_point else QColor(0, 0, 0))
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawEllipse(point, 5, 5)

class LassoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lasso Tool with Draggable Points")
        self.setGeometry(100, 100, 600, 400)
        self.lasso_widget = PolyLasso()
        self.setCentralWidget(self.lasso_widget)

if __name__ == "__main__":
    app = QApplication([])
    window = LassoApp()
    window.show()
    app.exec()