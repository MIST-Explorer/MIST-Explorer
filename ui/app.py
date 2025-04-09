from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import pandas as pd, tifffile as tiff, xml.etree.ElementTree as ET
from ui.toolbar.menubar_ui import MenuBarUI; from ui.toolbar.toolbar_ui import ToolBarUI; from ui.stardist.stardist_ui import StarDistUI; from ui.alignment.cell_intensity_ui import CellIntensityUI
from ui.processing.crop_ui import CropUI; from ui.processing.rotation_ui import RotateUI; from ui.canvas_ui import ImageGraphicsViewUI, ReferenceGraphicsViewUI
from ui.alignment.register_ui import RegisterUI
from ui.view_tab import ImageOverlay
from ui.analysis.AnalysisTab import AnalysisTab
from ui.processing.gaussian_blur import GaussianBlur
from core.canvas import MetaData

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
        
        self.resize(1280, 800)
        self.setMinimumSize(QSize(1024, 768))
        self.centralwidget = QWidget(self) # central widget inside main window
        self.central_widget_layout = QHBoxLayout(self.centralwidget) # add a layout to centralwidget so window can resize properly
        self.main_layout = QHBoxLayout() # main layout to add align canvas and the tab

        # add a menubar
        self.menubar = MenuBarUI(self, enc=self)

        # add a toolbar
        self.toolBar = ToolBarUI(self)
        
        # Create a container widget for the side panel
        self.sidePanel = QWidget(self.centralwidget)
        self.sidePanelLayout = QVBoxLayout(self.sidePanel)
        self.sidePanelLayout.setContentsMargins(0, 0, 0, 0)
        self.sidePanelLayout.setSpacing(0)
        self.sidePanel.setMaximumWidth(500)
        
        # Add collapse/expand button
        self.toggleButton = QPushButton("◀", self.sidePanel)
        self.toggleButton.setFixedSize(20, 60)
        self.toggleButton.clicked.connect(self.toggleSidePanel)
        self.toggleButton.setStyleSheet("""
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # Create a horizontal layout for the side panel and toggle button
        self.sidePanelContainer = QHBoxLayout()
        self.sidePanelContainer.setContentsMargins(0, 0, 0, 0)
        self.sidePanelContainer.setSpacing(0)
        
        # Create the stacked widget directly in the side panel
        self.stackedWidget = QStackedWidget(self.sidePanel)
        
        # canvas
        self.canvas = ImageGraphicsViewUI(self.centralwidget, enc=self)
        self.canvas.setMinimumSize(QSize(800, 500))

        ## images workspace#####
        images_scroll = QScrollArea()
        images_scroll.setWidgetResizable(True)
        images_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        images_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.images_tab = QWidget(self.canvas)
        self.images_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        images_scroll.setWidget(self.images_tab)
        self.stackedWidget.addWidget(images_scroll)

        ####### preprocess tab ###################################
        # Create scroll area for preprocessing tab
        preprocess_scroll = QScrollArea()
        preprocess_scroll.setWidgetResizable(True)
        preprocess_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        preprocess_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.preprocessing_tab = QWidget()
        self.horizontalLayout = QHBoxLayout(self.preprocessing_tab)
        self.preprocessing_dockwidget_main_vlayout = QVBoxLayout()
        self.horizontalLayout.addLayout(self.preprocessing_dockwidget_main_vlayout)
        
        # reference view
        self.small_view = ReferenceGraphicsViewUI(self.centralwidget)
        self.small_view.setParent(self.canvas)
        self.small_view.hide()

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
        # self.rotate_crop_hlayout.addWidget(self.gaussian_blur.gaussian_blur)
        self.rotate_crop_hlayout.setSpacing(3)
        
        self.preprocessing_dockwidget_main_vlayout.addLayout(self.rotate_crop_hlayout)
        self.register_groupbox = RegisterUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        self.gaussian_blur = GaussianBlur(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        # stardist UI
        self.stardist_groupbox = StarDistUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        self.cellIntensity_groupbox = CellIntensityUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.save_button)

        self.preprocessing_dockwidget_main_vlayout.setSpacing(5)
        self.preprocessing_dockwidget_main_vlayout.setContentsMargins(0, 0, 0, 0)

        preprocess_scroll.setWidget(self.preprocessing_tab)
        self.stackedWidget.addWidget(preprocess_scroll)

        ####### view tab #######################################
        # Create scroll area for view tab
        view_scroll = QScrollArea()
        view_scroll.setWidgetResizable(True)
        view_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.view_tab = ImageOverlay(self.canvas, enc=self)
        self.view_tab.setObjectName("view_tab")
        self.view_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        view_scroll.setWidget(self.view_tab)
        self.stackedWidget.addWidget(view_scroll)
        
        ####### analysis tab #######################################
        # Create scroll area for analysis tab
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.analysis_tab = AnalysisTab(self.canvas, self)
        self.analysis_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.analysis_tab.setObjectName("analysis_tab")
        analysis_scroll.setWidget(self.analysis_tab)
        self.stackedWidget.addWidget(analysis_scroll)


        metadata_scroll = QScrollArea()
        metadata_scroll.setWidgetResizable(True)
        metadata_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        metadata_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.metadata_tab = MetaData(self.canvas)
        self.metadata_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        metadata_scroll.setWidget(self.metadata_tab)
        self.stackedWidget.addWidget(metadata_scroll)



        # Set up the side panel layout
        self.sidePanelLayout.addWidget(self.stackedWidget)
        
        # Add the side panel and toggle button to the container
        self.sidePanelContainer.addWidget(self.sidePanel)
        self.sidePanelContainer.addWidget(self.toggleButton)
        
        # Add the container to the main layout
        self.main_layout.addLayout(self.sidePanelContainer)
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

        # Connect toolbar tab change signal
        self.toolBar.tabChanged.connect(self.stackedWidget.setCurrentIndex)
        # Start with Images tab
        self.stackedWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(self)
        

    def updateMousePositionLabel(self, text):
        self.toolBar.statusLine.setText(text)
    
    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MIST-Explorer"))

    saveSignal = pyqtSignal()
    def save_canvas(self):
        print("saving")
        self.saveSignal.emit()
        
    def update_progress_bar(self, value, str):
        if self.progressBar.value() == 100:
            self.progressBar.reset()
        self.progressBar.setValue(value)
        self.progressBarLabel.setText(str + " ...")

    def select(self):
        print("selecting")
        self.canvas.select = "rect"

    def circle_select(self):
        print("selecting")
        self.canvas.select = "circle"

    def poly_select(self):
        if self.canvas.select == "poly":
            self.canvas.select = False
            return
        
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

    def toggleSidePanel(self):
        if self.sidePanel.isVisible():
            self.sidePanel.hide()
            self.toggleButton.setText("▶")
        else:
            self.sidePanel.show()
            self.toggleButton.setText("◀")

    
    def get_metadata(self, metadata: dict):
        self.metadata = metadata
        self.metadata_tab.populate_table(self.metadata)
