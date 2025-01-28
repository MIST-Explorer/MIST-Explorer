from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QSpinBox, QPushButton, QWidget, QFileDialog
import pandas as pd, numpy as np, os
from PyQt6.QtCore import QCoreApplication, QMetaObject, pyqtSignal


class RegisterUI(QWidget):
    errorSignal = pyqtSignal(str)
    emitBeadData = pyqtSignal(np.ndarray)
    emitColorCode = pyqtSignal(pd.DataFrame)
    def __init__(self, parent=None, containing_layout:QVBoxLayout=None):
        super().__init__()
        self.setupUI(parent, containing_layout)


    def setupUI(self, parent, containing_layout:QVBoxLayout):
        self.register_groupbox = QGroupBox(parent)

        self.horizontalLayout_4 = QHBoxLayout(self.register_groupbox)
        self.register_components_vlayout = QVBoxLayout()
        self.register_components_vlayout.setSpacing(0)
        self.register_components_vlayout.setContentsMargins(0, 0, 0, 0)

        # ALIGNMENT LAYER
        self.hasblue_layout = QHBoxLayout()
        self.has_blue_color = QComboBox(self.register_groupbox)
        self.has_blue_color.addItems(["Yes", "No"])
        self.hasblue_label = QLabel()
        self.hasblue_layout.addWidget(self.hasblue_label)
        self.hasblue_layout.addWidget(self.has_blue_color)
        self.register_components_vlayout.addLayout(self.hasblue_layout)
        
        # ALIGNMENT LAYER
        self.alignment_layer_layout = QHBoxLayout()
        self.alignment_layer = QComboBox(self.register_groupbox)
        self.alignment_layer_label = QLabel()
        self.alignment_layer_layout.addWidget(self.alignment_layer_label)
        self.alignment_layer_layout.addWidget(self.alignment_layer)
        self.register_components_vlayout.addLayout(self.alignment_layer_layout)
  

        # cell layer
        self.protein_cell_layer_layout = QHBoxLayout()
        self.protein_cell_layer = QComboBox(self.register_groupbox)
        self.protein_cell_layer_label = QLabel()
        self.protein_cell_layer_layout.addWidget(self.protein_cell_layer_label)
        self.protein_cell_layer_layout.addWidget(self.protein_cell_layer)
        self.register_components_vlayout.addLayout(self.protein_cell_layer_layout)

        # intensity layer
        self.intensity_layer_layout = QHBoxLayout()
        self.intensity_layer = QComboBox(self.register_groupbox)
        self.intensity_layer_label = QLabel()
        self.intensity_layer_layout.addWidget(self.intensity_layer_label)
        self.intensity_layer_layout.addWidget(self.intensity_layer)
        self.register_components_vlayout.addLayout(self.intensity_layer_layout)

        # max size
        self.max_size_layout = QHBoxLayout()
        self.max_size_label =  QLabel(self.register_groupbox)
        self.max_size_layout.addWidget(self.max_size_label)
        self.max_size = QSpinBox(self.register_groupbox)
        self.max_size.setMaximum(100000)
        self.max_size.setSingleStep(100)
        self.max_size.setProperty("value", 23000)
        self.max_size_layout.addWidget(self.max_size)
        self.register_components_vlayout.addLayout(self.max_size_layout)

        # num tiles
        self.num_tiles_layout = QHBoxLayout()
        self.num_tiles_label =  QLabel(self.register_groupbox)
        self.num_tiles_layout.addWidget(self.num_tiles_label)
        self.num_tiles = QSpinBox(self.register_groupbox)
        self.num_tiles.setMaximum(30)
        self.num_tiles.setSingleStep(1)
        self.num_tiles.setProperty("value", 5)
        self.num_tiles_layout.addWidget(self.num_tiles)
        self.register_components_vlayout.addLayout(self.num_tiles_layout)


        # overlap
        self.overlap_layout = QHBoxLayout()
        self.overlap_label =  QLabel(self.register_groupbox)
        self.overlap_layout.addWidget(self.overlap_label)
        self.overlap = QSpinBox(self.register_groupbox)
        self.overlap.setMaximum(1000)
        self.overlap.setSingleStep(10)
        self.overlap.setProperty("value", 500)
        self.overlap_layout.addWidget(self.overlap)
        self.register_components_vlayout.addLayout(self.overlap_layout)

        # run button
        self.run_button = QPushButton(self.register_groupbox)
        self.register_components_vlayout.addWidget(self.run_button)

        # cancel button
        self.cancel_button = QPushButton(self.register_groupbox)
        self.register_components_vlayout.addWidget(self.cancel_button)
        self.horizontalLayout_4.addLayout(self.register_components_vlayout)
        containing_layout.addWidget(self.register_groupbox)

        self.__retranslate_UI()
        QMetaObject.connectSlotsByName(self)


    def updateChannelSelector(self, channels:dict):
        if self.alignment_layer.count() == 0 and self.protein_cell_layer.count() == 0 and self.intensity_layer.count() == 0:
            self.alignment_layer.addItems(list(channels.keys()))
            self.protein_cell_layer.addItems(list(channels.keys()))
            self.intensity_layer.addItems(list(channels.keys()))


    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.register_groupbox.setTitle(_translate("MainWindow", "Align Layers"))
        self.alignment_layer_label.setText(_translate("MainWindow", "Alignment Layer"))
        self.protein_cell_layer_label.setText(_translate("MainWindow", "Cell Layer"))
        self.intensity_layer_label.setText(_translate("MainWindow", "Protein Detection Layer"))
        self.max_size_label.setText(_translate("MainWindow", "Max Size"))
        self.num_tiles_label.setText(_translate("MainWindow", "Number of Tiles"))
        self.overlap_label.setText(_translate("MainWindow", "Overlap"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))
        self.hasblue_label.setText(_translate("MainWindow", "Align with Blue Color"))
