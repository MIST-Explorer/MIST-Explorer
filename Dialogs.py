from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QComboBox, QHBoxLayout, QGridLayout, QVBoxLayout, QCheckBox, QSlider, 
                             QListWidgetItem, QGraphicsView, QGraphicsScene, QListWidget, QPushButton, QLabel, QFileDialog,
                             QDialogButtonBox, QGraphicsPixmapItem)
from PyQt6.QtCore import Qt, QMetaObject, QCoreApplication
from PyQt6.QtGui import QPixmap, QPainter, QImage
import ui.app
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


class OpenFilesDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        # self.filename1 = None
        # self.filename2 = None
        # self.filename3 = None
        # self.filename4 = None


    def setupUi(self):
        self.setObjectName("Dialog")
        self.resize(270, 228)
        self.horizontalLayout = QHBoxLayout(self)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label1 = QLabel(self)
        self.label1.setMinimumSize(QSize(100, 0))
        self.label1.setObjectName("label1")
        self.horizontalLayout_3.addWidget(self.label1)
        self.button1 = QPushButton(self)
        self.button1.setObjectName("button1")
        self.horizontalLayout_3.addWidget(self.button1)
        self.filename1 = QLabel(self)
        self.filename1.setMinimumSize(QSize(100, 0))
        self.filename1.setObjectName("filename1")
        self.horizontalLayout_3.addWidget(self.filename1)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label2 = QLabel(self)
        self.label2.setMinimumSize(QSize(100, 0))
        self.label2.setObjectName("label2")
        self.horizontalLayout_4.addWidget(self.label2)
        self.button2 = QPushButton(self)
        self.button2.setObjectName("button2")
        self.horizontalLayout_4.addWidget(self.button2)
        self.filename2 = QLabel(self)
        self.filename2.setMinimumSize(QSize(100, 0))
        self.filename2.setObjectName("filename2")
        self.horizontalLayout_4.addWidget(self.filename2)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label3 = QLabel(self)
        self.label3.setMinimumSize(QSize(100, 0))
        self.label3.setObjectName("label3")
        self.horizontalLayout_5.addWidget(self.label3)
        self.button3 = QPushButton(self)
        self.button3.setObjectName("button3")
        self.horizontalLayout_5.addWidget(self.button3)
        self.filename3 = QLabel(self)
        self.filename3.setMinimumSize(QSize(100, 0))
        self.filename3.setObjectName("filename3")
        self.horizontalLayout_5.addWidget(self.filename3)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label4 = QLabel(self)
        self.label4.setMinimumSize(QSize(100, 0))
        self.label4.setObjectName("label4")
        self.horizontalLayout_6.addWidget(self.label4)
        self.button4 = QPushButton(self)
        self.button4.setObjectName("button4")
        self.horizontalLayout_6.addWidget(self.button4)
        self.filename4 = QLabel(self)
        self.filename4.setMinimumSize(QSize(100, 0))
        self.filename4.setObjectName("filename4")
        self.horizontalLayout_6.addWidget(self.filename4)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi()
        self.buttonBox.accepted.connect(self.accept) # type: ignore
        self.buttonBox.rejected.connect(self.reject) # type: ignore
        QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("self", "Select Files"))
        self.label1.setText(_translate("self", "Reference Image"))
        self.button1.setText(_translate("self", "Select File"))
        self.label2.setText(_translate("self", "Protein Image"))
        self.button2.setText(_translate("self", "Select File"))
        self.label3.setText(_translate("self", "Color Code"))
        self.button3.setText(_translate("self", "Select File"))
        self.label4.setText(_translate("self", "Decoding File"))
        self.button4.setText(_translate("self", "Select File"))


    def on_button1_pressed(self):
        filename1, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        self.filename1.setText(filename1)

    def on_button2_pressed(self):
        filename2, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        self.filename2.setText(filename2)

    def on_button3_pressed(self):
        filename3, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        self.filename3.setText(filename3)

    def on_button4_pressed(self):
        filename4, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        self.filename4.setText(filename4)


class ImageDialog(QDialog):
    def __init__(self, canvas, cropped_image:QPixmap):
        super().__init__()
        self.canvas = canvas
        self.cropped_image = cropped_image
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Cropped Image')

        self._layout = QVBoxLayout()

        self.image_view = QGraphicsView()

        self.image_scene = QGraphicsScene(self)  # Create a QGraphicsScene
        self.image_view.setScene(self.image_scene)  # Set the scene on the view
        self.cropped_pixmapItem = QGraphicsPixmapItem(self.cropped_image)
        self.image_view.scene().addItem(self.cropped_pixmapItem)
        self.image_view.setSceneRect(0, 0, 800, 600)
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
        self.canvas.updateCanvas(self.cropped_pixmapItem)
        self.accept()

    def cancel(self):
        self.confirm_crop = False
        self.reject()



# def numpy_to_qimage(array:np.ndarray):
#     if len(array.shape) == 2:
#         # Grayscale image
#         height, width = array.shape
#         qimage =  QImage(array.data, width, height, QImage.Format.Format_Grayscale8)
#     elif len(array.shape) == 3:
#         height, width, channels = array.shape
#         if channels == 3:
#             # RGB image
#             qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGB888)
#         elif channels == 4:
#             # RGBA image
#             qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGBA8888)
#     else:
#         raise ValueError("Unsupported array shape: {}".format(array.shape))
#     return qimage
