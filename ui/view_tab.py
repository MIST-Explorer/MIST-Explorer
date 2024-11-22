import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from stardist.models import StarDist2D
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import cv2
import os
from numba import njit
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

color_dict = {
    "red": [255, 0, 0],
    "blue": [0, 0, 255],
    "green": [0, 255, 0],
    "yellow": [255, 255, 0],
    "purple": [128, 0, 128],
    "orange": [255, 165, 0],
    "pink": [255, 192, 203],
    "brown": [165, 42, 42],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "silver": [192, 192, 192],
    "gray": [128, 128, 128],
    "maroon": [128, 0, 0],
    "olive": [128, 128, 0],
    "dark green": [0, 128, 0],
    "teal": [0, 128, 128],
    "navy": [0, 0, 128],
    "indigo": [75, 0, 130],
    "deep pink": [255, 20, 147],
    "chocolate": [210, 105, 30],
    "moccasin": [255, 228, 181],
    "steel blue": [70, 130, 180],
    "light sky blue": [135, 206, 250],
    "sandy brown": [244, 164, 96],
    "green yellow": [173, 255, 47],
    "khaki": [240, 230, 140],
    "tomato": [255, 99, 71],
    "dark slate blue": [72, 61, 139]
}


def rescale_array(arr, old_min, old_max, new_min, new_max):
    arr = np.array(arr)
    scale = (new_max - new_min) / (old_max - old_min)
    offset = new_min - old_min * scale
    rescaled_arr = arr * scale + offset

    return rescaled_arr

def winsorize_array(arr, lower_percentile, upper_percentile):
    arr = np.array(arr)
    lower_threshold = np.percentile(arr, lower_percentile * 100)
    upper_threshold = np.percentile(arr, upper_percentile * 100)
    winsorized_arr = np.clip(arr, lower_threshold, upper_threshold)
    
    return winsorized_arr


# def write_protein_sub(protein_data, reduced_cell_img):
#     cnv = reduced_cell_img.copy()
    
#     protein_1 = np.array(protein_data)
#     protein_1 = winsorize_array(protein_1, 0, .98)
#     protein_1 = rescale_array(protein_1, np.min(protein_1), np.max(protein_1), 60, 255)
    
#     print("prot written")

#     from tqdm import tqdm
#     for id, color in tqdm(enumerate(protein_1)):
#         # print("     one layer writte")
#         id += 1
#         cnv[reduced_cell_img == id] = color
        
#     return cnv 

import time
def write_protein(protein_data, reduced_cell_img):
    t = time.time()
    print(protein_data)
    cnv  = write_protein_sub(protein_data, reduced_cell_img)
    print(time.time() - t)
    
    return cnv

@njit
def write_protein_sub(protein_data=np.array([]), reduced_cell_img=np.array([[]])):
    # Copy the image
    cnv = reduced_cell_img.copy()

    # Extract protein data, winsorize, and rescale
    protein_1 = protein_data
    lower, upper = np.percentile(protein_1, [0, 100])
    protein_1 = np.clip(protein_1, lower, upper)
    #protein_1 = 60 + (protein_1 - lower) * (255 - 60) / (upper - lower)
    
    # Optimize the pixel update using vectorized operations
    for i in range(cnv.shape[0]):
        for j in range(cnv.shape[1]):
            id = reduced_cell_img[i, j]
            if id > 0 and id <= len(protein_1):
                cnv[i, j] = protein_1[id - 1]

    return cnv


def superimpose_image(base_shape, image, color):

    # Create a base array of zeros
    base = np.zeros(base_shape, dtype=np.uint8)
    
    # Ensure the image can be superimposed onto the base array
    if base_shape[:2] != image.shape:
        raise ValueError("Image shape must match the height and width of the base array.")
    
    # Set the color channels
    for i in range(3):
        base[:, :, i] = image * color[i]
    
    return base

def tint_grayscale_image(grayscale_image, color):
    """
    Tint a grayscale image with an arbitrary color.

    Parameters:
    grayscale_image (numpy array): The input grayscale image array with shape (height, width).
    color (tuple): The RGB color to use for tinting (R, G, B).

    Returns:
    numpy array: The tinted image with shape (height, width, 3).
    """
    # Ensure the grayscale image is a 2D array
    if len(grayscale_image.shape) != 2:
        raise ValueError("Input image must be a 2D array representing a grayscale image.")
    
    # Normalize the grayscale image to the range [0, 1]
    grayscale_image_normalized = grayscale_image / 255.0
    
    # Create an empty array with shape (height, width, 3) for the colored image
    tinted_image = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], 3), dtype=np.uint8)
    
    # Apply the color tint
    for i in range(3):
        tinted_image[:, :, i] = grayscale_image_normalized * color[i]
    
    return tinted_image

# ims = [write_protein(prot, ).astype("uint8") for prot in df.columns[3:]]

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, 
                             QGroupBox, QFormLayout, QScrollArea, QSizePolicy, QPushButton, 
                             QListWidget, QListWidgetItem, QDialog, QDialogButtonBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor

def scale_image_to_255(image_array):
    scaled_image = 255 * (image_array - image_array.min()) / (image_array.max() - image_array.min())
    return scaled_image.astype(np.uint8)


import sys, os
from PyQt6 import QtCore, QtGui, QtWidgets

__all__ = ['QRangeSlider']

DEFAULT_CSS = """



QRangeSlider * {
    border: 0px;
    padding: 0px;
    
}
QRangeSlider #Head {
    background: #222;
    
}
QRangeSlider #Span {
    background: #393;
    
}
QRangeSlider #Span:active {
    background: #282;
}
QRangeSlider #Tail {
    background: #222;
}
QRangeSlider > QSplitter::handle {
    background: #393;
}
QRangeSlider > QSplitter::handle:vertical {
    height: 4px;
}
QRangeSlider > QSplitter::handle:pressed {
    background: #ca5;
}
"""

def scale(val, src, dst):
    return int(((val - src[0]) / float(src[1]-src[0])) * (dst[1]-dst[0]) + dst[0])


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("QRangeSlider")
        Form.resize(300, 30)
        Form.setStyleSheet(DEFAULT_CSS)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self._splitter = QtWidgets.QSplitter(Form)
        self._splitter.setMinimumSize(QtCore.QSize(0, 0))
        self._splitter.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self._splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self._splitter.setObjectName("splitter")
        self._head = QtWidgets.QGroupBox(self._splitter)
        self._head.setTitle("")
        self._head.setObjectName("Head")
        self._handle = QtWidgets.QGroupBox(self._splitter)
        self._handle.setTitle("")
        self._handle.setObjectName("Span")
        self._tail = QtWidgets.QGroupBox(self._splitter)
        self._tail.setTitle("")
        self._tail.setObjectName("Tail")
        self.gridLayout.addWidget(self._splitter, 0, 0, 1, 1)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("QRangeSlider", "QRangeSlider"))


class Element(QtWidgets.QGroupBox):
    def __init__(self, parent, main):
        super(Element, self).__init__(parent)
        self.main = main

    def setStyleSheet(self, style):
        self.parent().setStyleSheet(style)

    def textColor(self):
        return getattr(self, '__textColor', QtGui.QColor(125, 125, 125))

    def setTextColor(self, color):
        if type(color) == tuple and len(color) == 3:
            color = QtGui.QColor(color[0], color[1], color[2])
        elif type(color) == int:
            color = QtGui.QColor(color, color, color)
        setattr(self, '__textColor', color)

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.main.drawValues():
            self.drawText(event, qp)
        qp.end()


class Head(Element):
    def __init__(self, parent, main):
        super(Head, self).__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QtGui.QFont('Arial', 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignmentFlag.AlignLeft, str(self.main.min()))


class Tail(Element):
    def __init__(self, parent, main):
        super(Tail, self).__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QtGui.QFont('Arial', 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignmentFlag.AlignRight, str(self.main.max()))


class Handle(Element):
    def __init__(self, parent, main):
        super(Handle, self).__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QtGui.QFont('Arial', 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignmentFlag.AlignLeft, str(self.main.start()))
        qp.drawText(event.rect(), QtCore.Qt.AlignmentFlag.AlignRight, str(self.main.end()))

    # def mouseMoveEvent(self, event):
    #     event.accept()
    #     mx = event.globalPosition().x()
    #     _mx = getattr(self, '__mx', None)
    #     if not _mx:
    #         setattr(self, '__mx', mx)
    #         dx = 0
    #     else:
    #         dx = mx - _mx
    #     setattr(self, '__mx', mx)
    #     if dx == 0:
    #         event.ignore()
    #         return
    #     elif dx > 0:
    #         dx = 1
    #     elif dx < 0:
    #         dx = -1
    #     s = self.main.start() + dx
    #     e = self.main.end() + dx
    #     if s >= self.main.min() and e <= self.main.max():
    #         self.main.setRange(s, e)
            
import qtrangeslider

class QRangeSlider(QtWidgets.QWidget, Ui_Form):
    endValueChanged = QtCore.pyqtSignal(int)
    maxValueChanged = QtCore.pyqtSignal(int)
    minValueChanged = QtCore.pyqtSignal(int)
    startValueChanged = QtCore.pyqtSignal(int)

    _SPLIT_START = 1
    _SPLIT_END = 2

    def __init__(self, parent=None):
        super(QRangeSlider, self).__init__(parent)
        self.setupUi(self)
        self.setMouseTracking(False)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._head_layout = QtWidgets.QHBoxLayout()
        self._head_layout.setSpacing(0)
        self._head_layout.setContentsMargins(0, 0, 0, 0)
        self._head.setLayout(self._head_layout)
        self.head = Head(self._head, main=self)
        self._head_layout.addWidget(self.head)
        self._handle_layout = QtWidgets.QHBoxLayout()
        self._handle_layout.setSpacing(0)
        self._handle_layout.setContentsMargins(40, 0, 40, 0)
        self._handle.setLayout(self._handle_layout)
        self.handle = Handle(self._handle, main=self)
        self.handle.setTextColor((150, 255, 150))
        self._handle_layout.addWidget(self.handle)
        self._tail_layout = QtWidgets.QHBoxLayout()
        self._tail_layout.setSpacing(0)
        self._tail_layout.setContentsMargins(0, 0, 0, 0)
        self._tail.setLayout(self._tail_layout)
        self.tail = Tail(self._tail, main=self)
        self._tail_layout.addWidget(self.tail)
        self.setMinimum(0)
        self.setMaximum(255)
        self.setStart(0)
        self.setEnd(255)
        self.setDrawValues(True)

    def min(self):
        return getattr(self, '__min', None)

    def max(self):
        return getattr(self, '__max', None)

    def setMinimum(self, value):
        setattr(self, '__min', value)
        print(self.min())
        self.minValueChanged.emit(value)

    def setMaximum(self, value):
        setattr(self, '__max', value)
        print(self.min())
        self.maxValueChanged.emit(value)

    def start(self):
        return getattr(self, '__start', None)

    def end(self):
        return getattr(self, '__end', None)

    def _setStart(self, value):
        setattr(self, '__start', value)
        self.startValueChanged.emit(value)
        print(self.start())

    def setStart(self, value):
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_START)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setStart(value)
        print(self.start())

    def _setEnd(self, value):
        setattr(self, '__end', value)
        print(self.end())
        self.endValueChanged.emit(value)

    def setEnd(self, value):
        print(self.end())
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_END)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setEnd(value)

    def drawValues(self):
        return getattr(self, '__drawValues', None)

    def setDrawValues(self, draw):
        setattr(self, '__drawValues', draw)

    def getRange(self):
        return (self.start(), self.end())

    def setRange(self, start, end):
        self.setStart(start)
        self.setEnd(end)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            s = self.start()-1
            e = self.end()-1
            
            print(s, e)
        elif key == QtCore.Qt.Key_Right:
            s = self.start()+1
            e = self.end()+1
            
            print(s, e)
        else:
            event.ignore()
            return
        event.accept()
        if s >= self.min() and e <= self.max():
            self.setRange(s, e)
            
            print(s, e)

    def setBackgroundStyle(self, style):
        self._tail.setStyleSheet(style)
        self._head.setStyleSheet(style)

    def setSpanStyle(self, style):
        self._handle.setStyleSheet(style)

    def _valueToPos(self, value):
        return scale(value, (self.min(), self.max()), (0, self.width()))

    def _posToValue(self, xpos):
        return scale(xpos, (0, self.width()), (self.min(), self.max()))

    def _handleMoveSplitter(self, xpos, index):
        hw = self._splitter.handleWidth()
        def _lockWidth(widget):
            width = widget.size().width()
            widget.setMinimumWidth(width)
            widget.setMaximumWidth(width)
        def _unlockWidth(widget):
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16777215)
        if index == self._SPLIT_START:
            v = self._posToValue(xpos)
            _lockWidth(self._tail)
            if v >= self.end():
                return
            offset = -20
            w = xpos + offset
            self._setStart(v)
        elif index == self._SPLIT_END:
            v = self._posToValue(xpos + 2 * hw)
            _lockWidth(self._head)
            if v <= self.start():
                return
            offset = -40
            w = self.width() - xpos + offset
            self._setEnd(v)
        _unlockWidth(self._tail)
        _unlockWidth(self._head)
        _unlockWidth(self._handle)

# if __name__ == '__main__':

#     app = QtWidgets.QApplication(sys.argv)
#     QtWidgets.QWidget()
    
#     
#     app.exec()

def build_range_slider():
    rs = QRangeSlider()
    rs.show()
    rs.setRange(10, 250)
    rs.setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222, stop:1 #333);')
    rs.handle.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282, stop:1 #393);')
    return rs 

class LayerDialog(QDialog):
    
    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Layer to Add")
        self.setGeometry(100, 100, 400, 300)
        
        self.layers = layers
        
        self.layout = QHBoxLayout()
        
        self.layer_list = QListWidget()
        for i, layer in enumerate(layers):
            item = QListWidgetItem(layer['name'])
            self.layer_list.addItem(item)
        self.layout.addWidget(self.layer_list)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
        
        self.setLayout(self.layout)
    
    def get_selected_layer_index(self): 
        print([item.text() for item in self.layer_list.selectedItems()])
        print([l["name"] for l in self.layers])
        print([l["name"] for l in self.layers].index([item.text() for item in self.layer_list.selectedItems()][0]))
        
        selected_items = self.layer_list.selectedItems()
        if selected_items:
            selected_index = [l["name"] for l in self.layers].index([item.text() for item in self.layer_list.selectedItems()][0])
            
            print("selected index", selected_index)
            return selected_index
        return None

class ColorDialog(QDialog):
    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Tint Color")
        self.setGeometry(100, 100, 400, 300)
        
        self.colors = colors
        
        self.layout = QVBoxLayout()
        
        self.color_list = QListWidget()
        for color_name in colors.keys():
            item = QListWidgetItem(color_name)
            self.color_list.addItem(item)
        self.layout.addWidget(self.color_list)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
        
        self.setLayout(self.layout)
    
    def get_selected_color_name(self):
        selected_items = self.color_list.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None


# changes made
def adjust_contrast(img, min=5, max = 100):
            # pixvals = np.array(img)
            minval = np.percentile(img, min) # room for experimentation 
            maxval = np.percentile(img, max) # room for experimentation 
            img = np.clip(img, minval, maxval)
            img = ((img - minval) / (maxval - minval)) * 255
            return (img.astype(np.uint8))
        
from PyQt6.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem

class ImageOverlay(QWidget):
    changePix = pyqtSignal(QGraphicsPixmapItem)
    
    def __init__(self, pixmap_label):
        super().__init__()
        
        self.pixmap_label = pixmap_label
        self.active_images = []
        
        self.df_path =  "/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
        self.im_path = "/Users/clark/Downloads/new_sd.png"

        self.loaded_df = None
        
        self.initUI()
        
    def load_stardist_image(self):
        if self.im_path == None:
            return None
        
        stardist_labels = Image.open(self.im_path)
        stardist_labels = np.array(stardist_labels)
        
        # return stardist_labels
        
        # reduced_cell_img = cv2.resize(stardist_labels.astype("float32"), (3000, 3000), interpolation = cv2.INTER_NEAREST_EXACT)
        
        scale_down_factor = 1 / self.scale_down.value()
        
        reduced_cell_img = cv2.resize(stardist_labels, (0,0), fx=scale_down_factor, fy=scale_down_factor, interpolation = cv2.INTER_NEAREST_EXACT) 
        return  reduced_cell_img
    
    
    def load_df(self):
        if self.df_path == None:
            return None
        
        if self.loaded_df is not None:
            return self.loaded_df
    
        
        print("df path", self.df_path)
        if self.df_path.endswith("csv"):
            df = pd.read_csv(self.df_path)
        if self.df_path.endswith("xlsx"):
            df = pd.read_excel(self.df_path)

        print("df raw", df)
        df = df[df.columns.drop(list(df.filter(regex='N/A')))]

        self.loaded_df = df
        return df
    
    
    def build_all(self):
        
        import time 
        
        if not hasattr(self, 'im_path'):

            import ui.app
            QMessageBox.critical(ui.app.Ui_MainWindow(), "Error", "Please an load image first!")
            return
        
        if not hasattr(self, 'df_path'):

            import ui.app
            QMessageBox.critical(ui.app.Ui_MainWindow(), "Error", "Please load data first!")
            return
        
        print('buidling all')
        reduced_cell_img = (self.load_stardist_image()).astype(np.uint16)
        df = self.load_df()
        self.df = df
        print(" df loaded")
        
        start = time.time()
        
        ims = [write_protein(np.array(df[protein_name]), np.array(reduced_cell_img)) for protein_name in df.columns[3:]]
        print("ims write", time.time() - start)
        ims = [adjust_contrast(im) for im in ims]  
        print("contrast adjusted") 
        ims = [tint_grayscale_image(ims[i], [255, 255, 255]) for i in range(len(ims))]
        print("tinted")
        layer_names = list(df.columns[3:]) 
        
        print("df", df)
        
        self.active_images_names = []
        self.ims = ims # List of images as np.arrays
        self.layer_names = layer_names  # List of layer names
        self.color_dict = color_dict  # Dictionary of color names to RGB values
        self.opacity_sliders = []
        self.contrast_sliders = []
        self.visibility_buttons = []
        self.color_tints = []
        self.color_labels = []
        self.group_box_array = []
        self.current_opacities = [1.0] * len(ims)
        self.current_contrasts = [1.0] * len(ims)
        self.current_visibilities = [True] * len(ims)
        self.current_tints = [QColor(255, 255, 255)] * len(ims)  # Default to no tint
        
        self.layers = [
            {'name': layer_names[i], 'image': ims[i]} 
            for i in range(len(ims))
        ] 
        
        # print(self.layers)
        
        print("done")
        
        self.scroll_area.setVisible(True)
        
        self.add_layer_button.setVisible(True)
        self.open_image.setVisible(False)
        self.open_image_label.setVisible(False)
        self.open_df.setVisible(False)
        self.open_df_label.setVisible(False)
        
        self.scale_down_label.setVisible(False)
        self.scale_down.setVisible(False)
        
        # self.todo_label.setVisible(False)
        
        self.apply_button.setVisible(False)
        self.cancel_reset.setVisible(True)
        
        return (ims, layer_names)
         
    def get_layer_values_at(self, x, y):
        if len(self.active_images) == 0:
            return None
        
        layer_values = []
        for name, img in zip(self.active_images_names, self.active_images):
            
            value = img[y, x]
            layer_values.append((name, value))
        return layer_values
    
    def get_layer_values_from_to(self, xstart, ystart, xend, yend):
        if len(self.active_images) == 0:
            return None

        layer_values = []

        for name, img in zip(self.active_images_names, self.active_images):
            values = img[ystart:yend, xstart:xend]
            layer_values.append((name, values))

        return layer_values
    
    def update_segmented_image(self, path):
        self.im_path = path
        
        
    def cancel_reset_first(self):
        reply = QMessageBox()
        reply.setText("Are you sure you want to reset?")
        reply.setStandardButtons(QMessageBox.StandardButton.Yes | 
                     QMessageBox.StandardButton.No)

        resp = reply.exec()

        if resp == QMessageBox.StandardButton.Yes:
            self.open_image.setVisible(True)
            self.open_image_label.setVisible(False)
            
            self.open_df.setVisible(True)
            self.open_df_label.setVisible(False)
            
            self.apply_button.setVisible(True)
            
            #
            
            self.add_layer_button.setVisible(False)
            self.cancel_reset.setVisible(False)
    
    
    def initUI(self):
        main_layout = QVBoxLayout()
        
        # print(self.image_label, "im label is")
        self.image_label = self.pixmap_label
        # print(self.image_label, "im label is")
        # main_layout.addWidget(self.image_label)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        
        
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        
        main_layout.addWidget(self.scroll_area)
        # self.scroll_area.addStretch()
        # self.scroll_area.setVisible(False)
        
        # - - - - -
        
        # self.todo_label = QLabel("To get started:\n     open a stardist image and its corresponding cell data\n     click apply!")
        # self.todo_label.setVisible(True)
        # main_layout.addWidget(self.todo_label)
        
        self.add_layer_button = QPushButton('Add Layer')
        self.add_layer_button.clicked.connect(self.show_layer_dialog)
        main_layout.addWidget(self.add_layer_button)
        self.add_layer_button.setVisible(False)
        
        self.cancel_reset = QPushButton('Cancel And Upload New')
        self.cancel_reset.clicked.connect(self.cancel_reset_first)
        main_layout.addWidget(self.cancel_reset)
        self.cancel_reset.setVisible(False)
        
        # self.r_u_sure = QPushButton('Are you sure?')
        # self.r_u_sure.clicked.connect(self.cancel_reset_second)
        # main_layout.addWidget(self.r_u_sure)
        # self.r_u_sure.setVisible(False)
        
        ## ----
        
        self.open_image = QPushButton('Open Image')
        self.open_image.clicked.connect(self.load_example)
        main_layout.addWidget(self.open_image)
        
        self.open_image_label = QLabel("Path: ")
        self.open_image_label.setVisible(False)
        main_layout.addWidget(self.open_image_label)
        
        self.open_df = QPushButton('Open Cell Data')
        self.open_df.clicked.connect(self.get_df_path)
        main_layout.addWidget(self.open_df)
        
        self.open_df_label = QLabel("Path: ")
        self.open_df_label.setVisible(False)
        main_layout.addWidget(self.open_df_label)
        
        ### scale slider
        self.scale_down_label = QLabel("Scale Down Factor: ")
        main_layout.addWidget(self.scale_down_label)
        
        self.scale_down =  QSlider(Qt.Orientation.Horizontal)
        self.scale_down.setTickPosition(QSlider.TicksAbove)
        self.scale_down.valueChanged.connect(self.scale_slider_update)
        
        self.scale_down.setRange(1, 10)
        self.scale_down.setValue(4)
        
        main_layout.addWidget(self.scale_down)
        ### scale slider
        
        self.apply_button = QPushButton('Apply')
        self.apply_button.clicked.connect(self.build_all)
        main_layout.addWidget(self.apply_button)
        # main_layout.addStretch()
        # main_layout.setDirection(QVBoxLayout.BottomToTop)
        
        # self.main_btn = QPushButton('run main')
        # self.main_btn.clicked.connect(self.main)
        # main_layout.addWidget(self.main_btn)
        
        
        self.setLayout(main_layout)
        self.update_image()
    
    def scale_slider_update(self, value):
        if value == 1:
            self.scale_down_label.setText(f'Scale Down Factor: original size')
            return
        self.scale_down_label.setText(f'Scale Down Factor: 1/{value} of original size')
        
    
    def less_than_15_chars(self, string):
        if len(string) > 50:
            return string[:49] + "..."
        
        return string
    
    def load_example(self):
        # print("Yo")

        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        if file_name:
            # combined_image = np.array(Image.open(file_name))
            
            self.open_image_label.setText(f"File: {self.less_than_15_chars(os.path.basename(file_name))}")
            self.open_image_label.setVisible(True)
            self.im_path = file_name

            # height, width, _ = combined_image.shape

            # q_image = QImage(combined_image.tobytes(), width, height, QImage.Format.Format_RGBA8888)
            # self.changePix.emit(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))
        
    def get_df_path(self):

        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Spreadsheets (*.csv *.xlsx);;All Files (*)")
        
        if file_name:
            
            # print()
            self.open_df_label.setText(f"File: {self.less_than_15_chars(os.path.basename(file_name))}")
            self.open_df_label.setVisible(True)
            self.df_path = file_name

            
    def addImage(self, pixmap):
        '''add a new image'''

        self.channels = None
        self.resetTransform()
        
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = (pixmap)
        
    def show_layer_dialog(self):
        if not hasattr(self, 'layers'):

            import ui.app

            QMessageBox.critical(ui.app.Ui_MainWindow(), "Error", "Empty canvas, please an load image first")
            return
        
        dialog = LayerDialog(self.layers, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_index = dialog.get_selected_layer_index()
            print("selected indexxxx is ", selected_index)
            if selected_index is not None:
                self.add_layer(selected_index)
    
    def show_color_dialog(self, idx):
        print(idx)
        dialog = ColorDialog(self.color_dict, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_color_name = dialog.get_selected_color_name()
            if selected_color_name:
                selected_color = self.color_dict[selected_color_name]
                print(f"Selected color: {selected_color_name} - {selected_color}")
                self.current_tints[idx] = QColor(*selected_color)
                self.color_labels[idx].setText(selected_color_name)
                self.update_image()
    
    def add_layer(self, index):
        new_img = self.layers[index]['image']
        new_name = self.layers[index]['name']
        
        # print(new_name)
        self.active_images_names.append(new_name)
        self.active_images.append(new_img)
        self.layer_names.append(new_name)
        self.current_opacities.append(1.0)
        self.current_contrasts.append(1.0)
        self.current_visibilities.append(True)
        self.current_tints.append(QColor(255, 255, 255))
        self.add_layer_controls(index)
        self.update_image()
        
    def delete_layer(self, index):
        print(index)
        
        print("before", len(self.active_images))
        self.active_images_names.pop(index)
        self.active_images.pop(index)
        print(len(self.active_images))
        self.current_opacities.pop(index)
        self.current_contrasts.pop(index)
        self.current_visibilities.pop(index)
        self.current_tints.pop(index)
        
        layer = self.group_box_array[index]
        self.scroll_layout.removeWidget(layer)
        layer.deleteLater()
        layer = None
        
        self.group_box_array.pop(index)
        self.update_image()
        
        if len(self.active_images) == 0:
            combined_image = np.zeros((10,10,10))
        
            height, width, _ = combined_image.shape
            bytes_per_line = 3

            q_image = QImage(combined_image.tobytes(), width, height, QImage.Format.Format_RGB888) # interesting image.tobytes() works well, maybe you don't need to do 
            
            self.changePix.emit(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))
        
    def add_layer_controls(self, idx):
        layer_id = idx
        idx = len(self.active_images) - 1
        print("add alyers controls", idx)
        group_box = QGroupBox(f"Layer {idx + 1}: {self.layer_names[layer_id]}")
        
        group_layout = QFormLayout()
        
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        opacity_slider.valueChanged.connect(lambda value: self.update_opacity(value, idx))
        self.opacity_sliders.append(opacity_slider)
        group_layout.addRow("Opacity:", opacity_slider)
        
        # contrast_slider = QSlider(Qt.Orientation.Horizontal)
        # 
        # contrast_slider.setValue(100)
        # 
        contrast_slider = qtrangeslider.QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        contrast_slider.valueChanged.connect(lambda value: self.update_contrast(value, idx))
        contrast_slider.setMaximum(255)
        contrast_slider.setValue((0, 255))
        contrast_slider.setDecimals(0)

        self.contrast_sliders.append(contrast_slider)
        group_layout.addRow("Contrast:", contrast_slider)
        
        visibility_button = QPushButton("Toggle Visibility")
        visibility_button.setCheckable(True)
        visibility_button.setChecked(True)
        visibility_button.toggled.connect(lambda checked: self.update_visibility(checked, idx))
        self.visibility_buttons.append(visibility_button)
        group_layout.addRow("Visibility:", visibility_button)
        
        color_button = QPushButton("Select Tint Color")
        color_button.clicked.connect(lambda : self.show_color_dialog(idx))
        self.color_tints.append(color_button)
        color_label = QLabel("None")
        self.color_labels.append(color_label)
        color_layout = QHBoxLayout()
        color_layout.addWidget(color_button)
        color_layout.addWidget(color_label)
        group_layout.addRow("Tint Color:", color_layout)
        
        # delete_button = QPushButton("Show Stats")
        # delete_button.clicked.connect(lambda: self.show_stats(self.layer_names[layer_id]))
        # # self.visibility_buttons.append(delete_button)
        # group_layout.addRow("Stats:", delete_button)
        
        delete_button = QPushButton("Delete Layer")
        delete_button.clicked.connect(lambda: self.delete_layer(len(self.active_images) - 1))
        # self.visibility_buttons.append(delete_button)
        group_layout.addRow("", delete_button)
        
        group_box.setLayout(group_layout)
        self.group_box_array.append(group_box)
        self.scroll_layout.addWidget(group_box)
    
    
    def update_opacity(self, value, idx):
        self.current_opacities[idx] = value / 100.0
        self.update_image()
    
    def update_contrast(self, value, idx):
        value = [int(value[0]), int(value[1])]
        print("contrast updated value", value, idx)
        self.current_contrasts[idx] = value
        self.update_image()
    
    def update_visibility(self, checked, idx):
        self.current_visibilities[idx] = checked
        self.update_image()
    
    def apply_tint(self, img, color):
        # print("trying to dye it ", color)
        # print(self.current_tints)
        tint_img = np.zeros_like(img)
        for c in range(3):
            tint_img[:, :, c] = img[:, :, c] * (color.getRgb()[c] / 255.0)
        return tint_img
    
    def adjust_contrast(self, img, min=5, max = 100):
            # pixvals = np.array(img)
            minval = np.percentile(img, min) # room for experimentation 
            maxval = np.percentile(img, max) # room for experimentation 
            img = np.clip(img, minval, maxval)
            img = ((img - minval) / (maxval - minval)) * 255
            return (img.astype(np.uint8))
    
    def slider_adjust_contrast(self, img, range):
        # print("slider_adjust_contrast", img.dtype, range)
        # # min, max = range
        # winsorized_arr = np.clip(img, min, max)
        # return winsorized_arr
        return adjust_contrast(img)
        
    def update_image(self):
        if len(self.active_images) == 0:
            return 

        combined_image = np.zeros_like(self.active_images[0], dtype=np.float32)
        
        for img, opacity, contrast, visible, tint in zip(self.active_images, self.current_opacities, self.current_contrasts, self.current_visibilities, self.current_tints):
            if visible:
                print("dollar eighty", contrast)    
                 
                adjusted_img = img * 1.0
                # adjusted_img = winsorize_array(adjusted_img, 0, 255)
                if type(contrast) == type([]):
                    adjusted_img = np.clip(adjusted_img, contrast[0], contrast[1])  
                    adjusted_img = scale_image_to_255(adjusted_img)
                # Clip values to
                adjusted_img = self.apply_tint(adjusted_img, tint)
                adjusted_img = np.clip(adjusted_img, 0, 255)  # Clip values to stay in the valid range
                combined_image += adjusted_img * opacity
        
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        
        height, width, _ = combined_image.shape
        bytes_per_line = 3

        q_image = QImage(combined_image.tobytes(), width, height, QImage.Format.Format_RGB888)
        
        self.changePix.emit(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))


colors_dict = {
    "red": [255, 0, 0],
    "blue": [0, 0, 255],
    "green": [0, 255, 0],
    "yellow": [255, 255, 0],
    "purple": [128, 0, 128],
    "orange": [255, 165, 0],
    "pink": [255, 192, 203],
    "brown": [165, 42, 42],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "silver": [192, 192, 192],
    "gray": [128, 128, 128],
    "maroon": [128, 0, 0],
    "olive": [128, 128, 0],
    "dark green": [0, 128, 0],
    "teal": [0, 128, 128],
    "navy": [0, 0, 128],
    "indigo": [75, 0, 130],
    "deep pink": [255, 20, 147],
    "chocolate": [210, 105, 30],
    "moccasin": [255, 228, 181],
    "steel blue": [70, 130, 180],
    "light sky blue": [135, 206, 250],
    "sandy brown": [244, 164, 96],
    "green yellow": [173, 255, 47],
    "khaki": [240, 230, 140],
    "tomato": [255, 99, 71],
    "dark slate blue": [72, 61, 139]
}

