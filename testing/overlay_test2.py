from PyQt6 import QtWidgets, QtGui, QtCore

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas()
        self.resultImage = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_ARGB32)

        # Load or create channel images
        self.firstImage = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_ARGB32)
        self.firstImage.fill(QtGui.QColor(255, 0, 0, 128))  # Red with some transparency
        self.secondImage = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_ARGB32)
        self.secondImage.fill(QtGui.QColor(0, 255, 0, 128))  # Green with some transparency
        self.thirdImage = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_ARGB32)
        self.thirdImage.fill(QtGui.QColor(0, 199, 255, 128))  # Blue with some transparency

        self.update_canvas()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_canvas(self):
        painter = QtGui.QPainter(self.resultImage)

        # Clear the result image with a transparent color
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.resultImage.rect(), QtCore.Qt.GlobalColor.transparent)

        # Draw each channel image in sequence
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        channel_images = [self.firstImage, self.secondImage, self.thirdImage]

        for image in channel_images:
            painter.drawImage(0, 0, image)

        # Optional: Fill the remaining transparent areas with white
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_DestinationOver)
        painter.fillRect(self.resultImage.rect(), QtCore.Qt.GlobalColor.white)

        painter.end()

        self.canvas.updateCanvas(QtGui.QPixmap.fromImage(self.resultImage))

class Canvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)

    def updateCanvas(self, pixmap):
        self.setPixmap(pixmap)
        self.repaint()  # Force repaint to update the canvas

app = QtWidgets.QApplication([])
window = MyWidget()
window.show()
app.exec()
