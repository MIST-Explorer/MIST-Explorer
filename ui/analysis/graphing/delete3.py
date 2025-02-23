import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QObject, QPointF, pyqtSignal

class PolygonSelection:
    def __init__(self):
        self.points = []
        self.color = QColor(0, 0, 255, 100)
        self.completed = False

    def add_point(self, point):
        print(f"Adding point: {point}")
        self.points.append(point)

    def draw(self, painter):
        if len(self.points) > 1:
            painter.setBrush(self.color)
            painter.setPen(QPen(Qt.GlobalColor.blue, 2))
            painter.drawPolygon(self.points)

class Canvas(QWidget):
    def __init__(self, pixmap_path):
        super().__init__()
        print(f"Initializing Canvas with image: {pixmap_path}")
        self.pixmap = QPixmap(pixmap_path)
        self.polygons = []
        self.current_polygon = None
        
        # Ensure widget can receive focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        print(f"Mouse press event at: {event.pos()}")
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.current_polygon:
                self.current_polygon = PolygonSelection()
            self.current_polygon.add_point(event.pos())
            self.update()

    def keyPressEvent(self, event):
        print(f"Key press event: {event.key()}")
        if event.key() == Qt.Key.Key_Return and self.current_polygon:
            print("Enter key pressed - finalizing polygon")
            self.current_polygon.completed = True
            self.polygons.append(self.current_polygon)
            self.current_polygon = None
            self.update()

    def paintEvent(self, event):
        print("Paint event triggered")
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)
        
        for polygon in self.polygons:
            polygon.draw(painter)
        
        if self.current_polygon:
            self.current_polygon.draw(painter)

class PolygonSelectorApp(QMainWindow):
    def __init__(self, pixmap_path):
        super().__init__()
        self.setWindowTitle("Multi-Polygon Selector")
        
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.canvas = Canvas(pixmap_path)
        layout.addWidget(self.canvas)

def main():
    app = QApplication(sys.argv)
    p = "/Users/clark/Desktop/wang/protein_visualization_app/sample_data/image.png"  # Replace with your image path
    selector = PolygonSelectorApp(p)
    selector.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    

#     selector = PolygonSelectorApp(p)
#     selector.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()