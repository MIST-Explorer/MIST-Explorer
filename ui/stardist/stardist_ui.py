from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QWidget
from PyQt6.QtCore import QSize, QCoreApplication, QMetaObject

class StarDistUI(QWidget):
    def __init__(self, parent=None, containing_layout:QVBoxLayout=None):
        super().__init__()
        self.setupUI(parent, containing_layout)

    def setupUI(self, parent, containing_layout:QVBoxLayout):

        self.stardist_groupbox = QGroupBox(parent)
        # self.stardist_groupbox.setMinimumSize(QSize(300, 400))
        # self.stardist_groupbox.setMaximumSize(QSize(500, 300))

        self.horizontalLayout_4 = QHBoxLayout(self.stardist_groupbox)
        self.stardist_components_vlayout = QVBoxLayout()
        self.stardist_components_vlayout.setSpacing(0)
        self.stardist_components_vlayout.setContentsMargins(0, 0, 0, 0)

        # channel selector
        self.stardist_channel_selector_layout = QHBoxLayout()
        self.stardist_channel_selector = QComboBox(self.stardist_groupbox)
        self.stardist_channel_selector_label = QLabel()
        self.stardist_channel_selector_label.setText("Select Channel")
        self.stardist_channel_selector_layout.addWidget(self.stardist_channel_selector_label)
        self.stardist_channel_selector_layout.addWidget(self.stardist_channel_selector)
        self.stardist_components_vlayout.addLayout(self.stardist_channel_selector_layout)

        # pretrained 2D Model
        self.stardist_hlayout1 = QHBoxLayout()
        self.stardist_label1 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout1.addWidget(self.stardist_label1)
        self.stardist_pretrained_models = QComboBox(self.stardist_groupbox)
        self.stardist_pretrained_models.addItems(["2D_versatile_fluo", "2D_paper_dsb2018", "2D_versatile_he"])
        self.stardist_hlayout1.addWidget(self.stardist_pretrained_models)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout1)

        # percentile low
        self.stardist_hlayout2 = QHBoxLayout()
        self.stardist_label2 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout2.addWidget(self.stardist_label2)
        self.percentile_low = QDoubleSpinBox(self.stardist_groupbox)
        self.percentile_low.setProperty("value", 3.0)
        self.stardist_hlayout2.addWidget(self.percentile_low)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout2)

        # percentile high
        self.stardist_hlayout3 = QHBoxLayout()
        self.stardist_label3 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout3.addWidget(self.stardist_label3)
        self.percentile_high = QDoubleSpinBox(self.stardist_groupbox)
        self.percentile_high.setProperty("value", 99.80)
        self.stardist_hlayout3.addWidget(self.percentile_high)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout3)

        # prob threshold
        self.stardist_hlayout4 = QHBoxLayout()
        self.stardist_label4 =  QLabel(self.stardist_groupbox)
        self.stardist_hlayout4.addWidget(self.stardist_label4)
        self.prob_threshold = QDoubleSpinBox(self.stardist_groupbox)
        self.prob_threshold.setProperty("value", 0.48)
        self.stardist_hlayout4.addWidget(self.prob_threshold)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout4)

        # nms threshold
        self.stardist_hlayout5 = QHBoxLayout()
        self.stardist_label5 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout5.addWidget(self.stardist_label5)
        self.nms_threshold = QDoubleSpinBox(self.stardist_groupbox)
        self.nms_threshold.setProperty("value", 0.3)
        self.stardist_hlayout5.addWidget(self.nms_threshold)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout5)

        # number of tiles
        self.stardist_hlayout6 = QHBoxLayout()
        self.stardist_label6 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout6.addWidget(self.stardist_label6)
        self.n_tiles = QSpinBox(self.stardist_groupbox)
        self.n_tiles.setProperty("value", 0)
        self.stardist_hlayout6.addWidget(self.n_tiles)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout6)

        # dilation
        self.stardist_hlayout7 = QHBoxLayout()
        self.stardist_label7 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout7.addWidget(self.stardist_label7)
        self.radius = QSpinBox(self.stardist_groupbox)
        self.radius.setProperty("value", 5)
        self.stardist_hlayout7.addWidget(self.radius)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout7)



        # run button
        self.stardist_run_button = QPushButton(self.stardist_groupbox)
        self.stardist_components_vlayout.addWidget(self.stardist_run_button)
  

        #cancel button
        self.cancel_button = QPushButton(self.stardist_groupbox)
        self.stardist_components_vlayout.addWidget(self.cancel_button)
        # save button
        self.save_button = QPushButton(self.stardist_groupbox)
        self.stardist_components_vlayout.addWidget(self.save_button)
        self.horizontalLayout_4.addLayout(self.stardist_components_vlayout)
        containing_layout.addWidget(self.stardist_groupbox)

        self.__retranslate_UI()
        QMetaObject.connectSlotsByName(self)


    def updateChannelSelector(self, channels:dict):
        if self.stardist_channel_selector.count() == 0:
            self.stardist_channel_selector.addItems(list(channels.keys()))

    def clearChannelSelector(self):
        self.stardist_channel_selector.clear()


    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.stardist_groupbox.setTitle(_translate("MainWindow", "StarDist Cell Segmentation"))
        self.stardist_label1.setText(_translate("MainWindow", "Pre-trained 2D Model"))
        self.stardist_label2.setText(_translate("MainWindow", "Percentile Low"))
        self.stardist_label3.setText(_translate("MainWindow", "Percentile High"))
        self.stardist_label4.setText(_translate("MainWindow", "Probability/ Score Threshold"))
        self.stardist_label5.setText(_translate("MainWindow", "Overlap Threshold"))
        self.stardist_label6.setText(_translate("MainWindow", "Number of Tiles"))
        self.stardist_label7.setText(_translate("MainWindow", "Radius"))
        self.stardist_run_button.setText(_translate("MainWindow", "Run"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))