from PyQt6.QtWidgets import QApplication, QRubberBand, QMainWindow
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtCore import QRect, QPoint, Qt, QSize

class CircularRubberBand(QRubberBand):
    def __init__(self, shape, parent=None):
        super().__init__(shape, parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.blue)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw a circle inside the current geometry
        rect = self.rect()
        painter.drawEllipse(rect)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Circular RubberBand Example")

        self.rubberband = None
        self.center = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.center = event.position().toPoint()
            if not self.rubberband:
                self.rubberband = CircularRubberBand(QRubberBand.Shape.Rectangle, self)
            self.rubberband.setGeometry(QRect(self.center, self.center))  # Initial zero-size rectangle
            self.rubberband.show()

    def mouseMoveEvent(self, event):
        if self.rubberband and not self.center.isNull():
            current = event.position().toPoint()
            radius = int((self.center - current).manhattanLength())  # Calculate the radius
            top_left = self.center - QPoint(radius, radius)
            size = 2 * radius
            self.rubberband.setGeometry(QRect(top_left, QSize(size, size)))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rubberband:
            self.rubberband.hide()
            self.rubberband = None  # Optionally keep it visible if needed

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
