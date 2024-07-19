from PyQt6.QtWidgets import QToolBar, QWidget, QComboBox
from PyQt6.QtCore import Qt, QCoreApplication, pyqtSignal
from PyQt6.QtGui import QPainter
from ui.tool import Action
import matplotlib as mpl

class ToolBarUI(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.toolbar = QToolBar(parent)
        parent.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.__createActions(parent)
        # self.__addCmaps()
        self.__addAllOps() #not used right now
        self.__addActions()
        self.__retranslateUI()


    def updateChannelSelector(self, channels:dict):
        if self.channelSelector.count() == 0:
            self.channelSelector.addItems(list(channels.keys()))

    def clearChannelSelector(self):
        self.channelSelector.clear()

    def __createActions(self, parent):
        self.actionRotate = Action(parent, "actionRotate", "icons/rotate-right.png")
        self.actionReset = Action(parent, "actionReset", "icons/home.png")
        self.actionOpenBrightnessContrast = Action(parent, "actionBC", "icons/brightness.png")
        self.operatorComboBox = QComboBox(parent)
        self.channelSelector = QComboBox(parent)
        self.cmapSelector = QComboBox(parent)

        self.operatorComboBox.setMinimumContentsLength(15)
        self.channelSelector.setMinimumWidth(100)

    # def __addCmaps(self):
        # cmap1 = mpl.colormaps['viridis']
        # cmap2 = mpl.colormaps['BuPu']
        # self.cmapSelector.addItem(cmap1)

        
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

