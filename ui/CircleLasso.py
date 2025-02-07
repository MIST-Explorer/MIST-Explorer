from PyQt6.QtWidgets import QRubberBand
from PyQt6.QtGui import QPainter, QPen



from PyQt6.QtGui import QColor
import random


class CircleLasso(QRubberBand):
    def __init__(self, shape, parent=None):
        super().__init__(shape, parent)
        self.fill = self.getRandomColor()

        self.color = QColor(*self.fill[0:3])

        self.f = QColor(*self.fill)
        self.filled = False
        self.dragging_threshold = 5
        self.mousePressPos = None
        self.mouseMovePos = None
        self.hello = False

    def getRandomColor(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 50)

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw a circle inside the current geometry
        rect = self.rect()
        painter.drawEllipse(rect)


    def mousePressEvent1(self, event):
    # if self.mousePressPos is not None:
        # print("from  AnalysisRubberBand mouse press event")
        self.mousePressPos = event.pos()                # global
        self.mouseMovePos = event.pos() - self.pos()    # local
        # super(AnalysisRubberBand, self).mousePressEvent(event)
        self.hello = True

    def mouseMoveEvent1(self, event):
        if self.mousePressPos is not None and self.hello:
            pos = event.pos()
            moved = pos - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Move when user drag window more than dragging_threshold
                diff = pos - self.mouseMovePos
                self.move(diff)
                self.mouseMovePos = pos - self.pos()
            super(CircularRubberBand, self).mouseMoveEvent(event)

    def mouseReleaseEvent1(self, event):
        # print("from  AnalysisRubberBand mouseReleaseEvent")
        self.hello = False
        if self.mousePressPos is not None:
            moved = event.pos() - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Do not call click event or so on
                event.ignore()
            self.mousePressPos = None
            self.mouseMovePos = None

