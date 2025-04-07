from PyQt6.QtWidgets import QToolBar, QWidget, QComboBox, QLabel, QSizePolicy, QPushButton
from PyQt6.QtCore import Qt, QCoreApplication, pyqtSignal, QSize, pyqtSlot
from PyQt6.QtGui import QPainter, QIcon, QImage, QPixmap
from ui.toolbar.Action import Action
import matplotlib.pyplot as plt, numpy as np, cv2
from PyQt6.QtCore import pyqtSignal
from qtrangeslider import QRangeSlider

class ToolBarUI(QWidget):

    # Add signals for tab changes
    tabChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__()
        self.toolbar = QToolBar(parent)
        parent.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.__createActions(parent)
        self.__addCmaps()
        self.__addActions()
        self.__retranslateUI()

    def updateChannelSelector(self, channels:dict, clear=False):
        '''Updates the dropdown to select channels. Clears the dropdown if new image is added'''
        if clear:
            self.clearChannelSelector()
            self.channelSelector.addItems(list(channels.keys()))
   

    def clearChannelSelector(self):
        '''Clears the channel selector dropdown'''
        self.channelSelector.clear()

    def __createActions(self, parent):
        # Create tab selection buttons
        self.imagesButton = QPushButton("Images")
        self.preprocessingButton = QPushButton("Preprocessing")
        self.viewButton = QPushButton("View")
        self.analysisButton = QPushButton("Analysis")
        self.metaDataButton = QPushButton("Details")
        # Style the buttons
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border: none;
                background: transparent;
                font-size: 12px;

            }
            QPushButton:hover {
                background: rgba(0, 0, 0, 0.1);
            }
            QPushButton:checked {
                border-bottom: 2px solid #007AFF;
                font-weight: Bold;
            }
        """

        self.imagesButton.setStyleSheet(button_style)
        self.preprocessingButton.setStyleSheet(button_style)
        self.viewButton.setStyleSheet(button_style)
        self.analysisButton.setStyleSheet(button_style)
        self.metaDataButton.setStyleSheet(button_style)
        # Make buttons checkable and exclusive
        self.imagesButton.setCheckable(True)
        self.preprocessingButton.setCheckable(True)
        self.viewButton.setCheckable(True) 
        self.analysisButton.setCheckable(True)
        self.metaDataButton.setCheckable(True)
        
        # Connect button signals
        self.imagesButton.clicked.connect(lambda: self.onTabButtonClicked(0))
        self.preprocessingButton.clicked.connect(lambda: self.onTabButtonClicked(1))
        self.viewButton.clicked.connect(lambda: self.onTabButtonClicked(2))
        self.analysisButton.clicked.connect(lambda: self.onTabButtonClicked(3))
        self.metaDataButton.clicked.connect(lambda: self.onTabButtonClicked(4))
        
        # Check imagesbutton by default
        self.imagesButton.setChecked(True)

        # Create other actions
        self.actionRotate = Action(parent, "actionRotate", "assets/icons/rotate-right.png")
        self.actionReset = Action(parent, "actionReset", "assets/icons/reset.png")
        
        self.actionOpenBrightnessContrast = Action(parent, "actionBC", "assets/icons/brightness.png")
        # self.operatorComboBox = QComboBox(parent)
        self.channelSelector = QComboBox(parent)
        self.channelSelector.currentIndexChanged.connect(self.on_channelSelector_currentIndexChanged) #try avoiding connect signal within the same class, but this will do for now
        
        self.statusLine = QLabel("Welcome! Please load an image to get started.")
        self.auto_contrast_button = QPushButton(self)
        self.auto_contrast_button.setText("Auto Contrast")
        self.cmapSelector = QComboBox(parent)
        # self.statusLine.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.cmapSelector.currentTextChanged.connect(self.on_cmapTextChanged) #try avoiding connect signal within the same class, but this will do for now

        self.contrastSlider = QRangeSlider()
        self.contrastSlider.setOrientation(Qt.Orientation.Horizontal)
        self.contrastSlider.setMaximumWidth(200)
        # self.operatorComboBox.setMinimumContentsLength(15)
        self.channelSelector.setMinimumWidth(100)
        # self.statusLine.setText("Welcome")
        # self.statusLine.setReadOnly(True)

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
        # Add tab buttons first
        self.toolbar.addWidget(self.imagesButton)
        self.toolbar.addWidget(self.preprocessingButton)
        self.toolbar.addWidget(self.viewButton)
        self.toolbar.addWidget(self.analysisButton)
        self.toolbar.addWidget(self.metaDataButton)
        
        # Add separator
        self.toolbar.addSeparator()
        
        # Add other actions
        self.toolbar.addAction(self.actionReset)
        # self.toolbar.addAction(self.actionOpenBrightnessContrast)
        # self.toolbar.addWidget(self.operatorComboBox)
        self.toolbar.addWidget(self.channelSelector)
        self.toolbar.addWidget(self.cmapSelector)        
        self.statusLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.statusLine.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.statusLine.setStyleSheet("""margin: 10px; """)
        self.toolbar.addWidget(self.statusLine)

        self.toolbar.addWidget(self.auto_contrast_button)
        self.toolbar.addWidget(self.contrastSlider)

    def update_contrast_slider(self, values):
        min, max = values

        self.contrastSlider.blockSignals(True)

        self.contrastSlider.setValue(values)
        self.contrastSlider.blockSignals(False)


    def update_cmap_selector(self, cmap_value):
        '''Updates the color map dropdown to the current color map of the image'''
        self.cmapSelector.setCurrentText(cmap_value)

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

    def onTabButtonClicked(self, index):
        # Uncheck other buttons
        buttons = [self.imagesButton, self.preprocessingButton, self.viewButton, self.analysisButton, self.metaDataButton]
        for i, button in enumerate(buttons):
            if i != index:
                button.setChecked(False)
        # Emit signal
        self.tabChanged.emit(index)

