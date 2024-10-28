from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QSpinBox, QPushButton, QWidget, QFileDialog
import pandas as pd, numpy as np, os
from PyQt6.QtCore import QSize, QCoreApplication, QMetaObject, pyqtSignal


class CellIntensityUI(QWidget):
    errorSignal = pyqtSignal(str)
    emitBeadData = pyqtSignal(np.ndarray)
    emitColorCode = pyqtSignal(pd.DataFrame)
    def __init__(self, parent=None, containing_layout:QVBoxLayout=None):
        super().__init__()
        self.setupUI(parent, containing_layout)


    def setupUI(self, parent, containing_layout:QVBoxLayout):
        self.cell_intensity_groupbox = QGroupBox(parent)
        # self.cell_intensity_groupbox.setMinimumSize(QSize(300, 400))
        # self.cell_intensity_groupbox.setMaximumSize(QSize(500, 300))


        self.cell_intensity_groupbox.setObjectName("cell_intensity_groupbox")
        self.horizontalLayout_4 = QHBoxLayout(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout = QVBoxLayout()
        self.cellintensity_components_vlayout.setSpacing(0)
        self.cellintensity_components_vlayout.setContentsMargins(0, 0, 0, 0)

        # bead data
        self.bead_data_layout = QHBoxLayout()
        self.bead_data = QPushButton(self.cell_intensity_groupbox)
        self.bead_data_label = QLabel()
        self.bead_data_layout.addWidget(self.bead_data)
        self.bead_data_layout.addWidget(self.bead_data_label)
        self.cellintensity_components_vlayout.addLayout(self.bead_data_layout)

        # color code
        self.color_code_layout = QHBoxLayout()
        self.color_code= QPushButton(self.cell_intensity_groupbox)
        self.color_code_label = QLabel()
        self.color_code_layout.addWidget(self.color_code)
        self.color_code_layout.addWidget(self.color_code_label)
        self.cellintensity_components_vlayout.addLayout(self.color_code_layout)

        # ALIGNMENT LAYER
        self.alignment_layer_layout = QHBoxLayout()
        self.alignment_layer = QComboBox(self.cell_intensity_groupbox)
        self.alignment_layer_label = QLabel()
        self.alignment_layer_layout.addWidget(self.alignment_layer_label)
        self.alignment_layer_layout.addWidget(self.alignment_layer)
        self.cellintensity_components_vlayout.addLayout(self.alignment_layer_layout)
  

        # cell layer
        self.protein_cell_layer_layout = QHBoxLayout()
        self.protein_cell_layer = QComboBox(self.cell_intensity_groupbox)
        self.protein_cell_layer_label = QLabel()
        self.protein_cell_layer_layout.addWidget(self.protein_cell_layer_label)
        self.protein_cell_layer_layout.addWidget(self.protein_cell_layer)
        self.cellintensity_components_vlayout.addLayout(self.protein_cell_layer_layout)

        # intensity layer
        self.intensity_layer_layout = QHBoxLayout()
        self.intensity_layer = QComboBox(self.cell_intensity_groupbox)
        self.intensity_layer_label = QLabel()
        self.intensity_layer_layout.addWidget(self.intensity_layer_label)
        self.intensity_layer_layout.addWidget(self.intensity_layer)
        self.cellintensity_components_vlayout.addLayout(self.intensity_layer_layout)

        # max size
        self.max_size_layout = QHBoxLayout()
        self.max_size_label =  QLabel(self.cell_intensity_groupbox)
        self.max_size_layout.addWidget(self.max_size_label)
        self.max_size = QSpinBox(self.cell_intensity_groupbox)
        self.max_size.setMaximum(100000)
        self.max_size.setSingleStep(100)
        self.max_size.setProperty("value", 23000)
        self.max_size_layout.addWidget(self.max_size)
        self.cellintensity_components_vlayout.addLayout(self.max_size_layout)

        # num tiles
        self.num_tiles_layout = QHBoxLayout()
        self.num_tiles_label =  QLabel(self.cell_intensity_groupbox)
        self.num_tiles_layout.addWidget(self.num_tiles_label)
        self.num_tiles = QSpinBox(self.cell_intensity_groupbox)
        self.num_tiles.setMaximum(10)
        self.num_tiles.setSingleStep(1)
        self.num_tiles.setProperty("value", 5)
        self.num_tiles_layout.addWidget(self.num_tiles)
        self.cellintensity_components_vlayout.addLayout(self.num_tiles_layout)


        # overlap
        self.overlap_layout = QHBoxLayout()
        self.overlap_label =  QLabel(self.cell_intensity_groupbox)
        self.overlap_layout.addWidget(self.overlap_label)
        self.overlap = QSpinBox(self.cell_intensity_groupbox)
        self.overlap.setMaximum(1000)
        self.overlap.setSingleStep(10)
        self.overlap.setProperty("value", 500)
        self.overlap_layout.addWidget(self.overlap)
        self.cellintensity_components_vlayout.addLayout(self.overlap_layout)


        # num cycles
        self.num_cycles_layout = QHBoxLayout()
        self.num_cycles_label =  QLabel(self.cell_intensity_groupbox)
        self.num_cycles_layout.addWidget(self.num_cycles_label)
        self.num_cycles = QSpinBox(self.cell_intensity_groupbox)
        self.num_cycles.setProperty("value", 3)
        self.num_cycles_layout.addWidget(self.num_cycles)
        self.cellintensity_components_vlayout.addLayout(self.num_cycles_layout)

        # num layers each
        self.num_layers_each_layout = QHBoxLayout()
        self.num_layers_each_label = QLabel(self.cell_intensity_groupbox)
        self.num_layers_each_layout.addWidget(self.num_layers_each_label)
        self.num_layers_each = QSpinBox(self.cell_intensity_groupbox)
        self.num_layers_each.setProperty("value", 3)
        self.num_layers_each_layout.addWidget(self.num_layers_each)
        self.cellintensity_components_vlayout.addLayout(self.num_layers_each_layout)

        # radius fg
        self.radius_fg_layout = QHBoxLayout()
        self.radius_fg_label = QLabel(self.cell_intensity_groupbox)
        self.radius_fg_layout.addWidget(self.radius_fg_label)
        self.radius_fg = QSpinBox(self.cell_intensity_groupbox)
        self.radius_fg.setProperty("value", 2)
        self.radius_fg_layout.addWidget(self.radius_fg)
        self.cellintensity_components_vlayout.addLayout(self.radius_fg_layout)

        # radius bg
        self.radius_bg_layout = QHBoxLayout()
        self.radius_bg_label = QLabel(self.cell_intensity_groupbox)
        self.radius_bg_layout.addWidget(self.radius_bg_label)
        self.radius_bg = QSpinBox(self.cell_intensity_groupbox)
        self.radius_bg.setProperty("value", 6)
        self.radius_bg_layout.addWidget(self.radius_bg)
        self.cellintensity_components_vlayout.addLayout(self.radius_bg_layout)

        # run button
        self.run_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.run_button)

        # cancel button
        self.cancel_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.cancel_button)

        # save button
        self.save_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.save_button)
        self.horizontalLayout_4.addLayout(self.cellintensity_components_vlayout)
        containing_layout.addWidget(self.cell_intensity_groupbox)

        self.__retranslate_UI()
        QMetaObject.connectSlotsByName(self)


    def loadBeadData(self):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Bead Data", "", "Bead Data(*.csv *.xlsx);;All Files (*)")
        if file_name:
            try:
                bead_data = pd.read_csv(file_name).to_numpy().astype("uint16")  # this is the output from the registration->decoding program
                self.emitBeadData.emit(bead_data)
                self.bead_data_label.setText(os.path.basename(file_name))
            except UnicodeDecodeError:
                self.errorSignal.emit("Please select a valid file type")

    def loadColorCode(self):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Color Code", "", "Color Code(*.csv *.xlsx);;All Files (*)")
        if file_name:
            try:
                color_code = pd.read_csv(file_name)
                self.emitColorCode.emit(color_code)
                self.color_code_label.setText(os.path.basename(file_name))
            except UnicodeDecodeError:
                self.errorSignal.emit("Please select a valid file type")


    def updateChannelSelector(self, channels:dict):
        if self.alignment_layer.count() == 0 and self.protein_cell_layer.count() == 0 and self.intensity_layer.count() == 0:
            self.alignment_layer.addItems(list(channels.keys()))
            self.protein_cell_layer.addItems(list(channels.keys()))
            self.intensity_layer.addItems(list(channels.keys()))


    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.cell_intensity_groupbox.setTitle(_translate("MainWindow", "Generate Protein Data of Cells"))
        self.alignment_layer_label.setText(_translate("MainWindow", "Alignment Layer"))
        self.protein_cell_layer_label.setText(_translate("MainWindow", "Cell Layer"))
        self.intensity_layer_label.setText(_translate("MainWindow", "Protein Detection Layer"))
        self.color_code_label.setText(_translate("MainWindow", "none selected"))
        self.color_code.setText(_translate("MainWindow", "Open Color Code"))
        self.bead_data_label.setText(_translate("MainWindow", "none selected"))
        self.bead_data.setText(_translate("MainWindow", "Open Bead Data"))
        self.radius_fg_label.setText(_translate("MainWindow", "Radius fg"))
        self.radius_bg_label.setText(_translate("MainWindow", "Radius bg"))
        self.max_size_label.setText(_translate("MainWindow", "Max Size"))
        self.num_tiles_label.setText(_translate("MainWindow", "Number of Tiles"))
        self.overlap_label.setText(_translate("MainWindow", "Overlap"))
        self.num_cycles_label.setText(_translate("MainWindow", "Number of decoding cycles"))
        self.num_layers_each_label.setText(_translate("MainWindow", "Number of decoding colors"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))