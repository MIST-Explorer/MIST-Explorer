from PyQt6.QtWidgets import QToolBar, QWidget, QComboBox
from PyQt6.QtCore import Qt, QCoreApplication, pyqtSignal, QSize, pyqtSlot
from PyQt6.QtGui import QPainter, QIcon, QImage, QPixmap
from ui.tool import Action
import matplotlib.pyplot as plt, numpy as np, cv2
from PyQt6.QtCore import pyqtSignal
from qtrangeslider import QRangeSlider

class ToolBarUI(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.toolbar = QToolBar(parent)
        parent.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.__createActions(parent)
        self.__addCmaps()
        # self.__addAllOps() #not used right now
        self.__addActions()
        self.__retranslateUI()

    def updateChannels(self, channels):
        self.np_channels = channels
    # def update_contrast(self, values):
    #     min_val, max_val = values  # Get the new min and max from slider
    #     contrast_image = self.apply_contrast(min_val, max_val)

    #     # self.label.setPixmap(self.numpy_to_pixmap(contrast_image))

    # def apply_contrast(self, new_min, new_max):
    #     # Get the actual min and max of the original image
    #     orig_min, orig_max = np.min(self.image), np.max(self.image)
        
    #     # Create a LUT based on the original and new min/max values
    #     lut = self.create_lut(orig_min, orig_max, new_min, new_max)
        
    #     # Apply the LUT to the image
    #     return cv2.LUT(self.image, lut)

    # def create_lut(self, orig_min, orig_max, new_min, new_max):
    #     lut = np.zeros(256, dtype=np.uint8)
    #     for i in range(256):
    #         if i < orig_min:
    #             lut[i] = new_min
    #         elif i > orig_max:
    #             lut[i] = new_max
    #         else:
    #             lut[i] = ((i - orig_min) / (orig_max - orig_min)) * (new_max - new_min) + new_min
    #     return lut

    def updateChannelSelector(self, channels:dict, clear=False):
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
        # self.operatorComboBox = QComboBox(parent)
        self.channelSelector = QComboBox(parent)
        self.channelSelector.currentIndexChanged.connect(self.on_channelSelector_currentIndexChanged) #try avoiding connect signal within the same class, but this will do for now
        self.cmapSelector = QComboBox(parent)
        self.cmapSelector.currentTextChanged.connect(self.on_cmapTextChanged) #try avoiding connect signal within the same class, but this will do for now

        self.contrastSlider = QRangeSlider()
        self.contrastSlider.setOrientation(Qt.Orientation.Horizontal)
        self.contrastSlider.setMaximumWidth(200)
        self.contrastSlider.setMaximum(255)
        self.contrastSlider.setMinimum(0)
        # self.operatorComboBox.setMinimumContentsLength(15)
        self.channelSelector.setMinimumWidth(100)

    def __generateCmapThumbnails(self) -> np.ndarray:
        self.cmap_names = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
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
        # self.toolbar.addAction(self.actionOpenBrightnessContrast)
        # self.toolbar.addWidget(self.operatorComboBox)
        self.toolbar.addWidget(self.channelSelector)
        self.toolbar.addWidget(self.cmapSelector)
        self.toolbar.addWidget(self.contrastSlider)

    # def __addOp(self, mode, name:str):
    #     self.operatorComboBox.addItem(name, mode)

    # def __addAllOps(self):
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_SourceOver, ("SourceOver"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationOver, ("DestinationOver"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Clear, ("Clear"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Source, ("Source"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Destination, ("Destination"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_SourceIn, ("SourceIn"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationIn, ("DestinationIn"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_SourceOut, ("SourceOut"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationOut, ("DestinationOut"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_SourceAtop, ("SourceAtop"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_DestinationAtop, ("DestinationAtop"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Xor, ("Xor"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Plus, ("Plus"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Multiply, ("Multiply"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Screen, ("Screen"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Overlay, ("Overlay"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Darken, ("Darken"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Lighten, ("Lighten"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_ColorDodge, ("ColorDodge"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_ColorBurn, ("ColorBurn"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_HardLight, ("HardLight"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_SoftLight, ("SoftLight"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Difference, ("Difference"))
    #     self.__addOp(QPainter.CompositionMode.CompositionMode_Exclusion, ("Exclusion"))

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.toolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.actionReset.setToolTip(_translate("MainWindow", "Reset Image"))
        self.channelSelector.setToolTip(_translate("MainWindow", "Select a channel"))
        # self.actionOpenBrightnessContrast.setText(_translate("MainWindow", "Brightness and Contrast"))
        # self.actionOpenBrightnessContrast.setToolTip(_translate("MainWindow", "Brightness and Contrast"))
    
    channelChanged = pyqtSignal(int)
    @pyqtSlot(int)
    def on_channelSelector_currentIndexChanged(self, index):
        if self.channelSelector.count() != 0:
            print("current index: ", self.channelSelector.currentIndex())
            # self.controllerSignal.emit(self.view.toolBar.channelSelector.currentIndex())
            self.channelChanged.emit(index)
            # channel_pixmap = QPixmap.fromImage(self.model_canvas.channels[self.view.toolBar.channelSelector.itemText(index)])
            # self.model_canvas.toPixmapItem(channel_pixmap)

    cmapChanged = pyqtSignal(str)
    @pyqtSlot(str)
    def on_cmapTextChanged(self, cmap_str :str):
        if self.channelSelector.count() != 0:
            print("current cmap str: ", self.cmapSelector.currentText())
            self.cmapChanged.emit(cmap_str)

