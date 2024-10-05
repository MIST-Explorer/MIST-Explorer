
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton, QWidget, QSlider
from PyQt6.QtCore import QSize, Qt, QCoreApplication


class ThresholdUI(QWidget):
    def __init__(self, parent=None, containing_layout:QVBoxLayout=None):
        super().__init__()
        self.thresholding_groupBox = QGroupBox(parent)
        self.thresholding_groupBox.setObjectName("thresholding_groupBox")
        horizontalLayout = QHBoxLayout(self.thresholding_groupBox)
        horizontalLayout.setObjectName("horizontalLayout_9")
        thresholding_components_vlayout = QVBoxLayout()

        
        self.thresholding_label1 = QLabel(self.thresholding_groupBox)
        self.thresholding_label1.setObjectName("thresholding_label1")


        self.channel_selector_layout = QHBoxLayout()
        self.channel_selector = QComboBox(self.thresholding_groupBox)
        self.channel_selector_label = QLabel()
        self.channel_selector_label.setText("Select Channel")
        self.channel_selector_layout.addWidget(self.channel_selector_label)
        self.channel_selector_layout.addWidget(self.channel_selector)
        thresholding_components_vlayout.addLayout(self.channel_selector_layout)
        thresholding_components_vlayout.addWidget(self.thresholding_label1)


        self.thresholding_slider1 = QSlider(self.thresholding_groupBox)
        self.thresholding_slider1.setMinimumSize(QSize(100, 0))
        self.thresholding_slider1.setOrientation(Qt.Orientation.Horizontal)
        self.thresholding_slider1.setObjectName("thresholding_slider1")
        thresholding_components_vlayout.addWidget(self.thresholding_slider1)
        self.thresholding_label2 = QLabel(self.thresholding_groupBox)
        self.thresholding_label2.setObjectName("thresholding_label2")
        thresholding_components_vlayout.addWidget(self.thresholding_label2)
        self.thresholding_slider2 = QSlider(self.thresholding_groupBox)
        self.thresholding_slider2.setMinimumSize(QSize(100, 0))
        self.thresholding_slider2.setOrientation(Qt.Orientation.Horizontal)
        self.thresholding_slider2.setObjectName("thresholding_slider2")
        thresholding_components_vlayout.addWidget(self.thresholding_slider2)
        self.thresholding_run_button = QPushButton(self.thresholding_groupBox)
        self.thresholding_run_button.setObjectName("thresholding_run_button")
        thresholding_components_vlayout.addWidget(self.thresholding_run_button)
        horizontalLayout.addLayout(thresholding_components_vlayout)
        containing_layout.addWidget(self.thresholding_groupBox)

        self.__retranslateUI()

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.thresholding_groupBox.setTitle(_translate("MainWindow", "Thresholding"))
        self.thresholding_label1.setText(_translate("MainWindow", "Block Size"))
        self.thresholding_label2.setText(_translate("MainWindow", "Offset"))
        self.thresholding_run_button.setText(_translate("MainWindow", "Confirm"))