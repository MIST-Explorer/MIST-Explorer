from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import pandas as pd

from ui.toolbar.menubar_ui import MenuBarUI; from ui.toolbar.toolbar_ui import ToolBarUI; from ui.stardist.stardist_ui import StarDistUI; from ui.alignment.cell_intensity_ui import CellIntensityUI
from ui.processing.crop_ui import CropUI; from ui.processing.rotation_ui import RotateUI; from ui.canvas_ui import ImageGraphicsViewUI, ReferenceGraphicsViewUI
from ui.alignment.register_ui import RegisterUI
from ui.view_tab import ImageOverlay
from ui.analysis.AnalysisTab import AnalysisTab
from ui.processing.gaussian_blur import GaussianBlur

class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QImageReader.setAllocationLimit(0)
        super().__init__()
        
        self.analysis_shortcut = QShortcut(QKeySequence("Ctrl+R"), self) # A for analysis
        self.analysis_shortcut.activated.connect(self.select)

        self.analysis_shortcut = QShortcut(QKeySequence("Ctrl+C"), self) # C for circle select
        self.analysis_shortcut.activated.connect(self.circle_select)

        self.analysis_shortcut = QShortcut(QKeySequence("Ctrl+P"), self) # C for circle select
        self.analysis_shortcut.activated.connect(self.poly_select)
        
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save)
        
        self.setupUI()

    def select(self):
        print("selecting")
        self.canvas.select = "rect"

    def circle_select(self):
        print("selecting")
        self.canvas.select = "circle"

    def poly_select(self):
        print("selecting")
        if self.canvas.select == "poly":
            self.canvas.select = False
        self.canvas.select = "poly"
    
    def save(self):
        from PIL import Image
        from PyQt6.QtWidgets import QFileDialog
        import numpy as np
        
        file_name, _ = QFileDialog.getSaveFileName(None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)")
        
        if file_name:
            pixmap = self.canvas.grab().toImage()
            buffer = pixmap.bits().asstring(pixmap.width() * pixmap.height() * pixmap.depth() // 8)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((pixmap.height(), pixmap.width(), pixmap.depth() // 8))
            if image.shape[2] == 4:  # If the image has an alpha channel
                image = image[:, :, :3]  # Remove the alpha channel
            image = image[:, :, ::-1]  # Convert BGR to RGB
            Image.fromarray(image).save(file_name)
        
    
    def setupUI(self):
        self.resize(1280, 800)
        self.setMinimumSize(QSize(1024, 768))
        self.centralwidget = QWidget(self) # central widget inside main window
        self.central_widget_layout = QHBoxLayout(self.centralwidget) # add a layout to centralwidget so window can resize properly
        self.main_layout = QHBoxLayout() # main layout to add align canvas and the tab

        # add a menubar
        self.menubar = MenuBarUI(self, enc=self)

        # add a toolbar
        self.toolBar = ToolBarUI(self)
        
        # initialize tabs 
        self.tabScrollArea = QScrollArea(self.centralwidget)
        self.tabScrollArea.setMinimumSize(QSize(400,500))
        self.tabScrollArea.setMaximumWidth(450)
        
        self.tabWidget = QTabWidget(self.tabScrollArea)

        
        # canvas
        self.canvas = ImageGraphicsViewUI(self.centralwidget, enc=self)
        self.canvas.setMinimumSize(QSize(800, 500))

        self.small_view = ReferenceGraphicsViewUI(self.centralwidget)
        self.small_view.setGeometry(520, 85, 200, 150)  # Position over the large view

        ####### preprocess tab ###################################
        
        self.preprocessUISetup() # uses self.canvas #stardist

        self.tabScrollArea.setWidget(self.tabWidget)
        self.tabScrollArea.setWidgetResizable(True)  
        self.tabScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.tabScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.tabWidget.addTab(self.preprocessing_tab, "Preprocessing")

        ####### view tab #######################################

        self.view_tab = ImageOverlay(self.canvas, enc=self)
        self.view_tab.setObjectName("view_tab")
        self.view_tab.setMaximumWidth(450)
        self.tabWidget.addTab(self.view_tab, "")
        
        ####### analysis tab #######################################

        self.analysis_tab = AnalysisTab(self.canvas, self)
        self.analysis_tab.setMaximumWidth(450)

        self.analysis_tab.setObjectName("analysis_tab")
        self.tabWidget.addTab(self.analysis_tab, "")
        # grid = QGridLayout(self)
        # grid.addWidget(self.tabWidget)
        self.setStyleSheet('''
        QTabWidget::tab-bar {
            alignment: left;
        }''')
        self.main_layout.addWidget(self.tabScrollArea)

        ##########################################################
        

        self.main_layout.addWidget(self.canvas) 
        self.central_widget_layout.addLayout(self.main_layout)
        self.setCentralWidget(self.centralwidget)
        
        # status bar
        container = QWidget()
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.progressBarLabel = QLabel("")
        self.progressBar = QProgressBar()
        progressBarLayout = QHBoxLayout()
        self.progressBar.setMaximum(100)
        progressBarLayout.addWidget(self.progressBarLabel)
        progressBarLayout.addWidget(self.progressBar)
        container.setLayout(progressBarLayout)

        self.statusbar.addPermanentWidget(container)
        
        progressBarStyle = """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 2px;
                text-align: right;
                height: 5px;
                margin-right: 30px;
            }
        
            QProgressBar::chunk {
                background-color: green;
                width: 20px;
            }
        """
        self.progressBar.setStyleSheet(progressBarStyle)
  
        self.progressBar.text
        self.__retranslateUI()

        self.tabWidget.setCurrentIndex(0) # start from preprocessing tab

        QMetaObject.connectSlotsByName(self)
        
    def onChange(self,i): #changed!
        if i == 0:
            print("preprocess selected")
        elif i == 1:
            print("view selected")

        elif i == 2:
            print("analysis selected")

    def updateMousePositionLabel(self, text):
        self.toolBar.statusLine.setText(text)
    
    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.preprocessing_tab), _translate("MainWindow", "Preprocessing"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.view_tab), _translate("MainWindow", "View"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.analysis_tab), _translate("MainWindow", "Analysis"))

    saveSignal = pyqtSignal()
    def save_canvas(self):
        
        print("saving")
        self.saveSignal.emit()
        
    def preprocessUISetup(self):
        self.preprocessing_tab = QWidget(self.tabWidget)
        self.preprocessing_tab.setMinimumHeight(1000)
        self.preprocessing_tab.setMaximumWidth(450)

        self.horizontalLayout = QHBoxLayout(self.preprocessing_tab)

        self.preprocessing_dockwidget_main_vlayout = QVBoxLayout()

        self.horizontalLayout.addLayout(self.preprocessing_dockwidget_main_vlayout) 

        self.save_button = QPushButton('Save Canvas')
        self.save_button.clicked.connect(self.save_canvas)
        
        # crop
        self.crop_groupbox = CropUI(self.preprocessing_tab)
        
        # rotate
        self.rotate_groupbox = RotateUI(self.preprocessing_tab)
        
        self.gaussian_blur = GaussianBlur(self.preprocessing_tab)
        
        # layout crop and rotate next to each other
        self.rotate_crop_hlayout = QHBoxLayout()
        
        self.rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        self.rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        self.rotate_crop_hlayout.addWidget(self.gaussian_blur.gaussian_blur)
        self.rotate_crop_hlayout.setSpacing(3)
        
        self.preprocessing_dockwidget_main_vlayout.addLayout(self.rotate_crop_hlayout)
        self.register_groupbox = RegisterUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        # stardist UI
        self.stardist_groupbox = StarDistUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        self.cellIntensity_groupbox = CellIntensityUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.save_button)

        self.preprocessing_dockwidget_main_vlayout.setSpacing(5)
        self.preprocessing_dockwidget_main_vlayout.setContentsMargins(0, 0, 0, 0)
        

    def updateProgressBar(self, value, str):
        if self.progressBar.value() == 100:
            self.progressBar.reset()
        self.progressBar.setValue(value)
        self.progressBarLabel.setText(str + " ...")

