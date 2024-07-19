from PyQt6.QtGui import QImageReader
from PyQt6.QtCore import QSize, QMetaObject, QCoreApplication
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QTabWidget, 
                             QStatusBar, QProgressBar, QGroupBox, QLabel)
# import protein_selection
import pandas as pd
from ui.menubar_ui import MenuBarUI; from ui.toolbar_ui import ToolBarUI; from ui.stardist_ui import StarDistUI; from ui.threshold_ui import ThresholdUI
from ui.crop_ui import CropUI; from ui.rotation_ui import RotateUI; from ui.canvas_ui import ImageGraphicsViewUI, ReferenceGraphicsViewUI
from ui.view_tab import ImageOverlay, color_dict, write_protein, adjust_contrast, tint_grayscale_image, load_stardist_image

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
        self.tabWidget = QTabWidget(self.tabScrollArea)
        self.tabWidget.setMinimumSize(QSize(400, 500))
        self.tabWidget.setMaximumSize(QSize(100, 2000))
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setObjectName("tabWidget")


        # canvas
        self.canvas = ImageGraphicsViewUI(self.centralwidget)
        self.canvas.setMinimumSize(QSize(800, 500))
        self.canvas.setObjectName("canvas")

        self.small_view = ReferenceGraphicsViewUI(self.centralwidget)
        self.small_view.setGeometry(520, 85, 200, 150)  # Position over the large view

        ####### preprocess tab ###################################
        self.preprocessUISetup() # uses self.canvas #stardist
        self.tabWidget.addTab(self.preprocessing_tab, "")
        
        ####### view tab #######################################
        reduced_cell_img = load_stardist_image()  
        # df = pd.read_csv("C:\\Users\\jianx\\protein_visualization_app\\sample_data\\celldta.csv")
        df = pd.read_csv("/Users/clark/Desktop/protein_visualization_app/sample_data/celldta.csv")
        df = df[df.columns.drop(list(df.filter(regex='N/A')))]
        print(df)
        ims = [write_protein(prot, reduced_cell_img).astype("uint8") for prot in df.columns[3:]]
        ims = [adjust_contrast(im) for im in ims]   
        ims = [tint_grayscale_image(ims[i], [255, 255, 255]) for i in range(len(ims))]
        layer_names = list(df.columns[3:])
        
        layers = [
            {'name': layer_names[i], 'image': ims[i]} 
            for i in range(len(ims))
        ]  # Example additional layers
        
        ims = ims[0:3]
        layer_names = layer_names[0:3]
        self.canvas.pixmapItem
        self.view_tab = ImageOverlay(self.canvas, ims, layer_names, color_dict)
        # self.canvas.updateCanvas()
        # self.view_tab = QLabel("hello")
        self.view_tab.setObjectName("view_tab")
        # self.protein_hlayout = QHBoxLayout(self.view_tab)
        # self.protein_hlayout.setObjectName("horizontalLayout_3")
        
        # add the protein selection boxes
        # self.proteinWidget_main_vlayout = QVBoxLayout()
        # self.proteinWidget_main_vlayout.setObjectName("proteinWidget_main_vlayout")
        # self.protein1_groupbox = protein_selection.Protein_Selector(self.view_tab)
        # self.protein2_groupbox = protein_selection.Protein_Selector(self.view_tab)
        # self.protein3_groupbox = protein_selection.Protein_Selector(self.view_tab)
        # self.proteinWidget_main_vlayout.addWidget(self.protein1_groupbox)
        # self.proteinWidget_main_vlayout.addWidget(self.protein2_groupbox)
        # self.proteinWidget_main_vlayout.addWidget(self.protein3_groupbox)
        # self.protein_hlayout.addLayout(self.proteinWidget_main_vlayout) # allow expansion of the groupbox when you resize
        
        self.tabWidget.addTab(self.view_tab, "")
        self.main_layout.addWidget(self.tabWidget)

        ########################################################

        self.main_layout.addWidget(self.canvas) 
        self.central_widget_layout.addLayout(self.main_layout)
        self.setCentralWidget(self.centralwidget)
        
        # status bar
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.progressBar = QProgressBar()
        self.progressBar.setMaximum(100)
        self.statusbar.addPermanentWidget(self.progressBar)
        self.progressBar.setValue(20)

        self.__retranslateUI()
        self.tabWidget.setCurrentIndex(0) # start from preprocessing tab

        QMetaObject.connectSlotsByName(self)

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.preprocessing_tab), _translate("MainWindow", "Preprocessing"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.view_tab), _translate("MainWindow", "View"))

    def preprocessUISetup(self):
        self.preprocessing_tab = QWidget()
        self.horizontalLayout = QHBoxLayout(self.preprocessing_tab)
        self.preprocessing_dockwidget_main_vlayout = QVBoxLayout()
        self.horizontalLayout.addLayout(self.preprocessing_dockwidget_main_vlayout) # from preprocessing tab

        # crop
        self.crop_groupbox = CropUI(self.preprocessing_tab)
        
        # rotate
        self.rotate_groupbox = RotateUI(self.preprocessing_tab)
        
        # layout crop and rotate next to each other
        self.rotate_crop_hlayout = QHBoxLayout()
        self.rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        self.rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        self.preprocessing_dockwidget_main_vlayout.addLayout(self.rotate_crop_hlayout)

        # thresholding UI
        self.threshold_groupbox = ThresholdUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        # stardist UI
        self.stardist_groupbox = StarDistUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        # registration UI
        self.registration_groupbox = QGroupBox()
        self.registration_groupbox.setTitle("Register")
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.registration_groupbox)