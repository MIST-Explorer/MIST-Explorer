from PyQt6.QtWidgets import QToolBar, QWidget, QComboBox
from PyQt6.QtCore import Qt, QCoreApplication, pyqtSignal, QSize, pyqtSlot
from PyQt6.QtGui import QPainter, QIcon, QImage, QPixmap
from ui.tool import Action
import matplotlib.pyplot as plt, numpy as np

class ToolBarUI(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.toolbar = QToolBar(parent)
        parent.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.__createActions(parent)
        self.__addCmaps()
        self.__addAllOps() #not used right now
        self.__addActions()
        self.__retranslateUI()

    def updateChannelSelector(self, channels:dict, _, clear=False):
        print("in toolbar, clearing?", clear)
        if clear:
            self.clearChannelSelector()
            self.channelSelector.addItems(list(channels.keys()))
        else:
            print("just updating, do nothing")

    def clearChannelSelector(self):
        print("toolbar channel selector is cleared")
        self.channelSelector.clear()

    def __createActions(self, parent):
        self.actionRotate = Action(parent, "actionRotate", "icons/rotate-right.png")
        self.actionReset = Action(parent, "actionReset", "icons/home.png")
        self.actionOpenBrightnessContrast = Action(parent, "actionBC", "icons/brightness.png")
        self.operatorComboBox = QComboBox(parent)
        self.channelSelector = QComboBox(parent)
        self.channelSelector.currentIndexChanged.connect(self.on_channelSelector_currentIndexChanged)
        self.cmapSelector = QComboBox(parent)

        self.operatorComboBox.setMinimumContentsLength(15)
        self.channelSelector.setMinimumWidth(100)

    def __generateCmapThumbnails(self) -> np.ndarray:
        self.cmap_names = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        self.cmap_thumbnail_arr = []
        for cmap_name in self.cmap_names:
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))

            # plot and save cmap as np array
            fig, ax= plt.subplots(figsize=(4,1))
            ax.imshow(gradient, aspect='auto', cmap=cmap_name)
            ax.set_xticks([])  
            ax.set_yticks([])  
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            fig.tight_layout(pad=0)

            fig.canvas.draw()
            thumbnail_arr = np.array(fig.canvas.renderer.buffer_rgba())
            self.cmap_thumbnail_arr.append(thumbnail_arr)
            plt.close()
        return self.cmap_thumbnail_arr

    def numpy_to_QIcon(self, array: np.ndarray):
        height, width, channels = array.shape
        image = QImage(array.tobytes(), width, height, QImage.Format.Format_RGBA8888)
        pixmap = QIcon(QPixmap.fromImage(image))
        return pixmap

    def __addCmaps(self):
        thumbnails = self.__generateCmapThumbnails()
        self.cmapSelector.setMinimumSize(QSize(100,20))
        for index, thumbnail in enumerate(thumbnails): 
            icon = self.numpy_to_QIcon(thumbnail)
            self.cmapSelector.setIconSize(QSize(100,20))
            self.cmapSelector.addItem(icon, self.cmap_names[index])
        
    def __addActions(self):
        self.toolbar.addAction(self.actionReset)
        self.toolbar.addAction(self.actionOpenBrightnessContrast)
        self.toolbar.addWidget(self.operatorComboBox)
        self.toolbar.addWidget(self.channelSelector)
        self.toolbar.addWidget(self.cmapSelector)

    def __addOp(self, mode, name:str):
        self.operatorComboBox.addItem(name, mode)

    def __addAllOps(self):
        self.__addOp(QPainter.CompositionMode.CompositionMode_SourceOver, ("SourceOver"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationOver, ("DestinationOver"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Clear, ("Clear"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Source, ("Source"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Destination, ("Destination"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_SourceIn, ("SourceIn"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationIn, ("DestinationIn"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_SourceOut, ("SourceOut"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationOut, ("DestinationOut"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_SourceAtop, ("SourceAtop"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationAtop, ("DestinationAtop"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Xor, ("Xor"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Plus, ("Plus"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Multiply, ("Multiply"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Screen, ("Screen"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Overlay, ("Overlay"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Darken, ("Darken"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Lighten, ("Lighten"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_ColorDodge, ("ColorDodge"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_ColorBurn, ("ColorBurn"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_HardLight, ("HardLight"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_SoftLight, ("SoftLight"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Difference, ("Difference"))
        self.__addOp(QPainter.CompositionMode.CompositionMode_Exclusion, ("Exclusion"))

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.toolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.actionReset.setToolTip(_translate("MainWindow", "Reset Image"))
        self.channelSelector.setToolTip(_translate("MainWindow", "Select a channel"))
        self.actionOpenBrightnessContrast.setText(_translate("MainWindow", "Brightness and Contrast"))
        self.actionOpenBrightnessContrast.setToolTip(_translate("MainWindow", "Brightness and Contrast"))
    
    channelChanged = pyqtSignal(int)
    @pyqtSlot(int)
    def on_channelSelector_currentIndexChanged(self, index):
        if self.channelSelector.count() != 0:
            print("current index: ", self.channelSelector.currentIndex())
            # self.controllerSignal.emit(self.view.toolBar.channelSelector.currentIndex())
            self.channelChanged.emit(index)
            # channel_pixmap = QPixmap.fromImage(self.model_canvas.channels[self.view.toolBar.channelSelector.itemText(index)])
            # self.model_canvas.toPixmapItem(channel_pixmap)

