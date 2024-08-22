from PyQt6 import QtWidgets, QtGui, QtCore


resultSize = QtCore.QSize(200,200)

class ImageComposer(QtWidgets.QWidget):

    def __init__(self):
                
        QtGui.QImageReader.setAllocationLimit(0)
        super().__init__()
        self.sourceButton = QtWidgets.QToolButton()
        self.sourceButton.setIconSize(resultSize)
        self.operatorComboBox = QtWidgets.QComboBox()
        self.sourceImage = QtGui.QImage()
        self.destinationImage = QtGui.QImage()
        self._resultImage = QtGui.QImage(resultSize, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver, ("SourceOver"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_DestinationOver, ("DestinationOver"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Clear, ("Clear"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Source, ("Source"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Destination, ("Destination"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_SourceIn, ("SourceIn"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_DestinationIn, ("DestinationIn"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_SourceOut, ("SourceOut"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_DestinationOut, ("DestinationOut"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_SourceAtop, ("SourceAtop"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_DestinationAtop, ("DestinationAtop"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Xor, ("Xor"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Plus, ("Plus"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Multiply, ("Multiply"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Screen, ("Screen"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Overlay, ("Overlay"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Darken, ("Darken"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Lighten, ("Lighten"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_ColorDodge, ("ColorDodge"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_ColorBurn, ("ColorBurn"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_HardLight, ("HardLight"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_SoftLight, ("SoftLight"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Difference, ("Difference"))
        self.addOp(QtGui.QPainter.CompositionMode.CompositionMode_Exclusion, ("Exclusion"))

        self.destinationButton = QtWidgets.QToolButton()
        self.destinationButton.setIconSize(resultSize)
        self.equalLabel = QtWidgets.QLabel("=")
        self.resultLabel = QtWidgets.QLabel()
        self.resultLabel.setMinimumWidth(resultSize.width())
# Connect signals and slots
        self.sourceButton.clicked.connect(self.chooseSource)
        self.operatorComboBox.activated.connect(self.recalculateResult)
        self.destinationButton.clicked.connect(self.chooseDestination)

        # Layout setup
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(self.sourceButton, 0, 0, 3, 1)
        mainLayout.addWidget(self.operatorComboBox, 1, 1)
        mainLayout.addWidget(self.destinationButton, 0, 2, 3, 1)
        mainLayout.addWidget(self.equalLabel, 1, 3)
        mainLayout.addWidget(self.resultLabel, 0, 4, 3, 1)
        mainLayout.setSizeConstraint(mainLayout.SizeConstraint.SetFixedSize)
        self.setLayout(mainLayout)

        # Initialize result image
        self.resultImage = QtGui.QImage(resultSize, QtGui.QImage.Format.Format_ARGB32_Premultiplied)

        # Load default images
        self.loadImage("butterfly.png", "source")
        self.loadImage("checker.png", "destination")

        self.setWindowTitle("Image Composition")

    def chooseSource(self):
        self.chooseImage("Choose Source Image", "source")

    def chooseDestination(self):
        self.chooseImage("Choose Destination Image", "destination")

    def recalculateResult(self):
        mode = self.currentMode()

        painter = QtGui.QPainter(self.resultImage)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.resultImage.rect(), QtCore.Qt.GlobalColor.transparent)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(0, 0, self.destinationImage)
        painter.setCompositionMode(mode)
        painter.drawImage(0, 0, self.sourceImage)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_DestinationOver)
        painter.fillRect(self.resultImage.rect(), QtCore.Qt.GlobalColor.white)
        painter.end()

        self.resultLabel.setPixmap(QtGui.QPixmap.fromImage(self.resultImage))

    def addOp(self, mode, name):
        self.operatorComboBox.addItem(name, mode)

    def chooseImage(self, title, imageType):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, title)
        if fileName:
            self.loadImage(fileName, imageType)

    def loadImage(self, fileName, imageType):
        if imageType == "source":
            self.sourceImage = QtGui.QImage(fileName)
            self.sourceImage = self.sourceImage.scaled(resultSize, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.sourceImage = self.fixImage(self.sourceImage)
            self.sourceButton.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(self.sourceImage)))
        elif imageType == "destination":
            self.destinationImage = QtGui.QImage(fileName)
            self.destinationImage = self.destinationImage.scaled(resultSize, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.destinationImage = self.fixImage(self.destinationImage)
            self.destinationButton.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(self.destinationImage)))

        self.recalculateResult()

    def fixImage(self, image):
        fixedImage = QtGui.QImage(resultSize, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        painter = QtGui.QPainter(fixedImage)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(fixedImage.rect(), QtCore.Qt.GlobalColor.transparent)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(self.imagePos(image), image)
        painter.end()
        return fixedImage

    def currentMode(self):
        return QtGui.QPainter.CompositionMode(self.operatorComboBox.itemData(self.operatorComboBox.currentIndex()))

    def imagePos(self, image:QtGui.QImage):
        return QtCore.QPoint((resultSize.width() - image.width()) // 2, (resultSize.height() - image.height()) // 2)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    composer = ImageComposer()
    composer.show()
    sys.exit(app.exec())
