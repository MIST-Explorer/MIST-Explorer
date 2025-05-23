from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QComboBox, QHBoxLayout, QGridLayout, QVBoxLayout, QCheckBox, QSlider, 
                             QListWidgetItem, QGraphicsView, QGraphicsScene, QListWidget, QPushButton, QLabel, QFileDialog,
                             QDialogButtonBox, QGraphicsPixmapItem)
from PyQt6.QtCore import Qt, QMetaObject, QCoreApplication
from PyQt6.QtGui import QPixmap, QPainter, QImage
import ui.app, numpy as np
from utils import numpy_to_qimage, scale_adjust
from cv2 import LUT
# classes for dialog pop-ups

class BrightnessContrastDialog(QDialog):
    def __init__(self, parent=None, channels:dict=None, canvas = None, operatorComboBox: QComboBox = None):
        super().__init__(parent)
        self.channels= channels
        self.canvas = canvas
        self.operatorComboBox = operatorComboBox
        self.setupUI()
        self.updateChannels()
        self.operatorComboBox.activated.connect(self.reCalculateResult)

    def setupUI(self):
        self.setObjectName("BrightnessContrastDialog")
        self.resize(700, 1000)
        self.setMinimumSize(QSize(500, 750))
        self.setMaximumSize(QSize(500, 750))
        self.dialog_resize_layout = QHBoxLayout(self)
        self.dialog_resize_layout.setObjectName("dialog_resize_layout")
        self.main_layout = QVBoxLayout()
        self.main_layout.setObjectName("main_layout")
        self.channel_list_widget = QListWidget(self)
        self.channel_list_widget.setObjectName("channel_list_widget")
        self.main_layout.addWidget(self.channel_list_widget)
        self.checkbox_layout = QGridLayout()
        self.checkbox_layout.setHorizontalSpacing(6)
        self.checkbox_layout.setObjectName("checkbox_layout")
        self.invert_background_checkbox = QCheckBox(self)
        self.invert_background_checkbox.setObjectName("invert_background_checkbox")
        self.checkbox_layout.addWidget(self.invert_background_checkbox, 0, 1, 1, 1)
        self.show_grayscale_checkbox = QCheckBox(self)
        self.show_grayscale_checkbox.setObjectName("show_grayscale_checkbox")
        self.checkbox_layout.addWidget(self.show_grayscale_checkbox, 0, 0, 1, 1)
        self.keep_settings_checkbox = QCheckBox(self)
        self.keep_settings_checkbox.setObjectName("keep_settings_checkbox")
        self.checkbox_layout.addWidget(self.keep_settings_checkbox, 1, 0, 1, 1)
        self.main_layout.addLayout(self.checkbox_layout)
        self.contrast_min_slider_layout = QHBoxLayout()
        self.contrast_min_slider_layout.setObjectName("contrast_min_slider_layout")
        self.contrast_min_label = QLabel(self)
        self.contrast_min_label.setMinimumSize(QSize(80, 0))
        self.contrast_min_label.setObjectName("contrast_min_label")
        self.contrast_min_slider_layout.addWidget(self.contrast_min_label)
        self.contrast_min_slider = QSlider(self)
        self.contrast_min_slider.setOrientation(Qt.Orientation.Horizontal)
        self.contrast_min_slider.setObjectName("contrast_min_slider")
        self.contrast_min_slider_layout.addWidget(self.contrast_min_slider)
        self.contrast_min_value_label = QLabel(self)
        self.contrast_min_value_label.setMinimumSize(QSize(80, 0))
        self.contrast_min_value_label.setObjectName("contrast_min_value_label")
        self.contrast_min_slider_layout.addWidget(self.contrast_min_value_label)
        self.main_layout.addLayout(self.contrast_min_slider_layout)
        self.contrast_max_slider_layout = QHBoxLayout()
        self.contrast_max_slider_layout.setObjectName("contrast_max_slider_layout")
        self.contrast_max_label = QLabel(self)
        self.contrast_max_label.setMinimumSize(QSize(80, 0))
        self.contrast_max_label.setObjectName("contrast_max_label")
        self.contrast_max_slider_layout.addWidget(self.contrast_max_label)
        self.contrast_max_slider = QSlider(self)
        self.contrast_max_slider.setOrientation(Qt.Orientation.Horizontal)
        self.contrast_max_slider.setObjectName("contrast_max_slider")
        self.contrast_max_slider_layout.addWidget(self.contrast_max_slider)
        self.contrast_max_value_label = QLabel(self)
        self.contrast_max_value_label.setMinimumSize(QSize(80, 0))
        self.contrast_max_value_label.setObjectName("contrast_max_value_label")
        self.contrast_max_slider_layout.addWidget(self.contrast_max_value_label)
        self.main_layout.addLayout(self.contrast_max_slider_layout)
        self.gamma_slider_layout = QHBoxLayout()
        self.gamma_slider_layout.setObjectName("gamma_slider_layout")
        self.gamma_label = QLabel(self)
        self.gamma_label.setMinimumSize(QSize(80, 0))
        self.gamma_label.setObjectName("gamma_label")
        self.gamma_slider_layout.addWidget(self.gamma_label)
        self.gamma_slider = QSlider(self)
        self.gamma_slider.setOrientation(Qt.Orientation.Horizontal)
        self.gamma_slider.setObjectName("gamma_slider")
        self.gamma_slider_layout.addWidget(self.gamma_slider)
        self.gamma_value_label = QLabel(self)
        self.gamma_value_label.setMinimumSize(QSize(80, 0))
        self.gamma_value_label.setObjectName("gamma_value_label")
        self.gamma_slider_layout.addWidget(self.gamma_value_label)
        self.main_layout.addLayout(self.gamma_slider_layout)
        self.auto_reset_layout = QHBoxLayout()
        self.auto_reset_layout.setObjectName("auto_reset_layout")
        self.auto_button = QPushButton(self)
        self.auto_button.setObjectName("auto_button")
        self.auto_reset_layout.addWidget(self.auto_button)
        self.reset_button = QPushButton(self)
        self.reset_button.setObjectName("reset_button")
        self.auto_reset_layout.addWidget(self.reset_button)
        self.main_layout.addLayout(self.auto_reset_layout)
        self.contrast_histogram_view = QGraphicsView(self)
        self.contrast_histogram_view.setAcceptDrops(False)
        self.contrast_histogram_view.setObjectName("contrast_histogram_view")
        self.main_layout.addWidget(self.contrast_histogram_view)
        self.dialog_resize_layout.addLayout(self.main_layout)

        self.retranslateUi()
        QMetaObject.connectSlotsByName(self)
        self.show()


    def reCalculateResult(self):
        '''calculate resulting image when changing how channels are overlaid'''
        mode = self.currentMode()

        if self.channels != None:

            painter = QPainter(self.canvas.resultImage)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(self.canvas.resultImage.rect(), Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            # painter.drawImage(0, 0, self.channels[.text()])
            painter.setCompositionMode(mode)
            # painter.drawImage(0, 0, self.channels[item.text()])
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
            painter.fillRect(self.canvas.resultImage.rect(), Qt.GlobalColor.white)
            painter.end()
            self.canvas.updateCanvas(QPixmap.fromImage(self.canvas.resultImage))

    def currentMode(self):
        return QPainter.CompositionMode(self.operatorComboBox.itemData(self.operatorComboBox.currentIndex()))
    

    def updateChannels(self):
        '''loads the channels in a tiff image'''
        if self.channels is not None:
            for channel_name in self.channels.keys():
                item = QListWidgetItem(channel_name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)  # Make the item checkable
                item.setCheckState(Qt.CheckState.Checked)  # Set initial state to checked
                self.channel_list_widget.addItem(item)

            self.channel_list_widget.itemClicked.connect(self.on_channel_clicked)


    def on_channel_clicked(self, item: QListWidgetItem):
        mode = self.currentMode()
        item.setSelected(True)

        # call reCalculateResult when the number of channels active are changed
        painter = QPainter(self.canvas.resultImage)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.canvas.resultImage.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(mode)

        for index in range(self.channel_list_widget.count()):
            _item = self.channel_list_widget.item(index)
            if _item.checkState() == Qt.CheckState.Checked:
                painter.drawImage(0,0, self.channels[_item.text()])
        painter.end()
        self.canvas.updateCanvas(QPixmap.fromImage(self.canvas.resultImage))
                
        # if (item.checkState() == Qt.CheckState.Checked):
        #     print(item.text())
        #     self.canvas.updateCanvas(self.channels[item.text()])
        #     # self.reCalculateResult()
        # elif (item.checkState() == Qt.CheckState.Unchecked):
        #     self.canvas.deleteImage()
            
            
    
    def on_gamma_slider_valueChanged(self, value):
        self.gamma_value_label.setText(str(value))

    def on_contrast_max_slider_valueChanged(self, value):
        self.contrast_max_value_label.setText(str(value))

    def on_contrast_min_slider_valueChanged(self, value):
        self.contrast_min_value_label.setText(str(value))

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("self", "Brightness and Contrast"))
        self.invert_background_checkbox.setText(_translate("self", "Invert background"))
        self.show_grayscale_checkbox.setText(_translate("self", "Show Grayscale"))
        self.keep_settings_checkbox.setText(_translate("self", "Keep Settings"))
        self.contrast_min_label.setText(_translate("self", "Min display"))
        self.contrast_max_label.setText(_translate("self", "Max display"))
        self.gamma_label.setText(_translate("self", "Gamma"))
        self.auto_button.setText(_translate("self", "Auto"))
        self.reset_button.setText(_translate("self", "Reset"))


class ImageDialog(QDialog):
    '''Popup window to confirm the cropped image'''
    def __init__(self, canvas, cropped_image:np.ndarray, contrast:tuple, cmap:str):
        super().__init__()
        self.canvas = canvas
        self.cropped_image = cropped_image
        self.contrast = contrast
        self.cmap = cmap
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Cropped Image')

        self._layout = QVBoxLayout()

        self.image_view = QGraphicsView()

        self.image_scene = QGraphicsScene(self)  # Create a QGraphicsScene
        self.image_view.setScene(self.image_scene)  # Set the scene on the view

        if self.cropped_image.dtype != np.uint8:
            self.cropped_image = scale_adjust(self.cropped_image)

        im = self.apply_contrast(self.cropped_image, self.contrast[0], self.contrast[1])
        self.pix = QPixmap(numpy_to_qimage(im))
        self.cropped_pixmapItem = QGraphicsPixmapItem(self.pix)
        self.image_view.scene().addItem(self.cropped_pixmapItem)
        self.image_view.setSceneRect(0, 0, self.pix.width(), self.pix.height())
        item_rect = self.cropped_pixmapItem.boundingRect()
        self.image_view.setSceneRect(item_rect)
        self.image_view.fitInView(self.cropped_pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_view.centerOn(self.cropped_pixmapItem)
        self._layout.addWidget(self.image_view)

        # Add buttons
        self.button_layout = QHBoxLayout()
        
        self.confirm_button = QPushButton('Confirm', self)
        self.confirm_button.clicked.connect(self.confirm)
        self.button_layout.addWidget(self.confirm_button)
        
        self.reject_button = QPushButton('Reject', self)
        self.reject_button.clicked.connect(self.cancel)
        self.button_layout.addWidget(self.reject_button)

        self._layout.addLayout(self.button_layout)

        self.setLayout(self._layout)

    def confirm(self):
        self.confirm_crop = True
        self.canvas.toPixmapItem(self.pix)
        self.accept()

    def cancel(self):
        self.confirm_crop = False
        self.reject()


    def apply_contrast(self, image, new_min, new_max):

        lut = self.create_lut(new_min, new_max)
        return LUT(image, lut)

    def create_lut(self, new_min, new_max):

        lut = np.zeros(256, dtype=np.uint8) # uint8 for display
        lut[new_min:new_max+1] = np.linspace(start=0, stop=255, num=(new_max - new_min + 1), endpoint=True, dtype=np.uint8)
        lut[:new_min] = 0 # clip between 0 and 255
        lut[new_max+1:] = 255

        return lut