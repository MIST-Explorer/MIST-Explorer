from PyQt6.QtGui import QImageReader

from PyQt6.QtCore import QSize, QMetaObject, QCoreApplication, Qt, pyqtSignal

from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QTabWidget, QStatusBar, QProgressBar, QGroupBox, QLabel, QPushButton, QSizePolicy)

import pandas as pd
from ui.menubar_ui import MenuBarUI; from ui.toolbar_ui import ToolBarUI; from ui.stardist_ui import StarDistUI; from ui.cell_intensity_ui import CellIntensityUI
from ui.crop_ui import CropUI; from ui.rotation_ui import RotateUI; from ui.canvas_ui import ImageGraphicsViewUI, ReferenceGraphicsViewUI
from ui.view_tab import ImageOverlay

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        QImageReader.setAllocationLimit(0)
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.resize(1280, 800)
        self.setMinimumSize(QSize(1024, 768))
        self.centralwidget = QWidget(self) # central widget inside main window
        self.central_widget_layout = QHBoxLayout(self.centralwidget) # add a layout to centralwidget so window can resize properly
        self.main_layout = QHBoxLayout() # main layout to add align canvas and the tab

        # add a menubar
        self.menubar = MenuBarUI(self)

        # add a toolbar
        self.toolBar = ToolBarUI(self)
        
        # initialize tabs        
        self.tabScrollArea = QScrollArea(self.centralwidget)
        self.tabScrollArea.setMinimumSize(QSize(400,600))
        self.tabScrollArea.setMaximumWidth(500)
        
        self.tabWidget = QTabWidget(self.tabScrollArea)
        self.tabWidget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.tabWidget.setMinimumSize(QSize(400, 800))
        # self.tabWidget.setMaximumSize(QSize(500, 1500))  

        self.tabScrollArea.setWidget(self.tabWidget)
        
        self.tabScrollArea.setWidgetResizable(True)  # make the scroll area resize with the widget
        self.tabScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tabScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)


        # canvas
        self.canvas = ImageGraphicsViewUI(self.centralwidget, enc=self)
        self.canvas.setMinimumSize(QSize(800, 500))

        self.small_view = ReferenceGraphicsViewUI(self.centralwidget)
        self.small_view.setGeometry(520, 85, 200, 150)  # Position over the large view

        ####### preprocess tab ###################################
        
        self.preprocessUISetup() # uses self.canvas #stardist
        self.tabWidget.addTab(self.preprocessing_tab, "")
        
        ####### view tab #######################################

        self.view_tab = ImageOverlay(self.canvas)
        self.view_tab.setObjectName("view_tab")
        self.tabWidget.addTab(self.view_tab, "")
        self.main_layout.addWidget(self.tabScrollArea)
        
        ########################################################

        self.main_layout.addWidget(self.canvas) 
        self.central_widget_layout.addLayout(self.main_layout)
        self.setCentralWidget(self.centralwidget)
        
        # status bar
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.progressBar = QProgressBar()
        self.statusbar.addPermanentWidget(self.progressBar)
        self.progressBar.setMaximum(100)

        self.__retranslateUI()

        self.tabWidget.setCurrentIndex(0) # start from preprocessing tab

        QMetaObject.connectSlotsByName(self)
        
    def onChange(self,i): #changed!
        if i == 0:
            self.tabWidget.setMinimumSize(QSize(400, 800))
        elif i == 1:
            self.tabWidget.setMinimumSize(QSize(400, 600))
            print("view selected")
    
    def updateMousePositionLabel(self, text):
        self.toolBar.statusLine.setText(text)
    
    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.preprocessing_tab), _translate("MainWindow", "Preprocessing"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.view_tab), _translate("MainWindow", "View"))

    saveSignal = pyqtSignal()
    def save_canvas(self):
        
        print("saving")
        self.saveSignal.emit()
        
    def preprocessUISetup(self):
        self.scrollarea = QScrollArea()
        self.preprocessing_tab = QWidget(self.scrollarea)
        self.horizontalLayout = QHBoxLayout(self.preprocessing_tab)
        # 

        self.preprocessing_dockwidget_main_vlayout = QVBoxLayout()

        self.horizontalLayout.addLayout(self.preprocessing_dockwidget_main_vlayout) 

        self.save_button = QPushButton('Save Canvas')
        self.save_button.clicked.connect(self.save_canvas)
        
        # crop
        self.crop_groupbox = CropUI(self.preprocessing_tab)
        
        # rotate
        self.rotate_groupbox = RotateUI(self.preprocessing_tab)
        
        
        
        # layout crop and rotate next to each other
        self.rotate_crop_hlayout = QHBoxLayout()
        
        self.rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        self.rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        
        
        self.preprocessing_dockwidget_main_vlayout.addLayout(self.rotate_crop_hlayout)
        
        # stardist UI
        self.stardist_groupbox = StarDistUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        self.cellIntensity_groupbox = CellIntensityUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.save_button)

        self.preprocessing_dockwidget_main_vlayout.setSpacing(5)
        self.preprocessing_dockwidget_main_vlayout.setContentsMargins(0, 0, 0, 0)
        

        

    def updateProgressBar(self, value):
        if self.progressBar.value() == 100:
            self.progressBar.reset()
        self.progressBar.setValue(value)

