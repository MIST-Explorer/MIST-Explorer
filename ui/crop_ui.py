from PyQt6.QtWidgets import  QHBoxLayout, QGroupBox, QToolButton, QWidget
from PyQt6.QtCore import QSize, QCoreApplication
from ui.Action import Action

class CropUI(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.crop_groupbox = QGroupBox(parent)
        self.crop_groupbox.setTitle("Crop")
        self.crop_groupbox.setMinimumWidth(100)
        self.crop_groupbox.setMaximumHeight(80)
        self.crop_button = QToolButton(self.crop_groupbox)
        cropAction = Action(self.crop_groupbox, "cropAction", "icons/crop.png")
        self.crop_button.setDefaultAction(cropAction)
        self.crop_button.setIconSize(QSize(40,40))

        self.cancel_crop_button = QToolButton(self.crop_groupbox)
        cancel_cropAction = Action(self.crop_groupbox, "cancelCropAction", "icons/cross.png")
        self.cancel_crop_button.setDefaultAction(cancel_cropAction)
        self.cancel_crop_button.setIconSize(QSize(40,40))

        croplayout = QHBoxLayout(self.crop_groupbox)
        croplayout.addWidget(self.crop_button)
        croplayout.addWidget(self.cancel_crop_button)

        self.__retranslateUI()

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.crop_button.setToolTip(_translate("MainWindow", "Crop Image"))
        self.cancel_crop_button.setToolTip(_translate("MainWindow", "Cancel"))