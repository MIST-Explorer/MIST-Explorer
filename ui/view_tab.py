import pandas as pd
import numpy as np
from stardist.models import StarDist2D
from PIL import Image

import cv2

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

def main():
    reduced_cell_img = load_stardist_image()  
    df = pd.read_csv("/Users/clark/Desktop/protein_visualization_app/sample_data/celldta.csv")
    ims = [write_protein(prot).astype("uint8") for prot in df.columns[3:]]
    ims = [adjust_contrast(im) for im in ims]   
    ims = [tint_grayscale_image(ims[i], [255, 255, 255]) for i in range(len(ims))]
    layer_names = list(df.columns[3:])
    
    layers = [
        {'name': layer_names[i], 'image': ims[i]} 
        for i in range(len(ims))
    ]  # Example additional layers
    
    ims = ims[0:3]
    layer_names = layer_names[0:3]


    app = QApplication(sys.argv)
    window = ImageOverlay(ims, layer_names, colors_dict)
    window.layers = layers  # Pass additional layers to the window
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

def load_stardist_image():
    stardist_labels = Image.open("/Users/clark/Downloads/protein_visualization_app/ttest/al/stardist_labels.png")
    stardist_labels = np.array(stardist_labels)
    
    reduced_cell_img = cv2.resize(stardist_labels.astype("float32"), (500, 500))
    
    return reduced_cell_img

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

df = pd.read_csv("/Users/clark/Desktop/protein_visualization_app/sample_data/celldta.csv")
df = df[df.columns.drop(list(df.filter(regex='N/A')))]
print(df.columns)

def write_protein(protein_name, reduced_cell_img):
    cnv = reduced_cell_img.copy()
    
    protein_1 = np.array(df[protein_name])
    protein_1 = winsorize_array(protein_1, 0, .98)
    protein_1 = rescale_array(protein_1, np.min(protein_1), np.max(protein_1), 60, 255)

    for id, color in enumerate(protein_1):
        id += 1
        cnv[reduced_cell_img == id] = color
        
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
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
        
        self.setLayout(self.layout)
    
    def get_selected_layer_index(self):
        selected_items = self.layer_list.selectedItems()
        if selected_items:
            selected_index = self.layer_list.row(selected_items[0])
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
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
        
        self.setLayout(self.layout)
    
    def get_selected_color_name(self):
        selected_items = self.color_list.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None

def adjust_contrast(img, min=5, max = 98):
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
        
        reduced_cell_img = load_stardist_image()  
        # df = pd.read_csv("C:\\Users\\jianx\\protein_visualization_app\\sample_data\\celldta.csv")
        df = pd.read_csv("/Users/clark/Desktop/protein_visualization_app/sample_data/celldta.csv")
        df = df[df.columns.drop(list(df.filter(regex='N/A')))]

        ims = [write_protein(prot, reduced_cell_img).astype("uint8") for prot in df.columns[3:]]
        ims = [adjust_contrast(im) for im in ims]   
        ims = [tint_grayscale_image(ims[i], [255, 255, 255]) for i in range(len(ims))]
        
        layer_names = list(df.columns[3:])
        
        
        self.ims = ims  # List of images as np.arrays
        self.layer_names = layer_names  # List of layer names
        self.color_dict = color_dict  # Dictionary of color names to RGB values
        self.opacity_sliders = []
        self.contrast_sliders = []
        self.visibility_buttons = []
        self.color_tints = []
        self.color_labels = []
        self.current_opacities = [1.0] * len(ims)
        self.current_contrasts = [1.0] * len(ims)
        self.current_visibilities = [True] * len(ims)
        self.current_tints = [QColor(255, 255, 255)] * len(ims)  # Default to no tint
        
        self.pixmap_label = pixmap_label
        
        self.layers = [
            {'name': layer_names[i], 'image': ims[i]} 
            for i in range(len(ims))
        ] 
        
        self.initUI()
        
    def initUI(self):
        main_layout = QVBoxLayout()
        
        # print(self.image_label, "im label is")
        self.image_label = self.pixmap_label
        print(self.image_label, "im label is")
        # main_layout.addWidget(self.image_label)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        
        for i, _ in enumerate(self.ims):
            self.add_layer_controls(i)
        
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        
        main_layout.addWidget(self.scroll_area)
        
        self.add_layer_button = QPushButton('Add Layer')
        self.add_layer_button.clicked.connect(self.show_layer_dialog)
        main_layout.addWidget(self.add_layer_button)
        
        self.add_example = QPushButton('Add example')
        self.add_example.clicked.connect(self.load_example)
        main_layout.addWidget(self.add_example)
        
        # self.main_btn = QPushButton('run main')
        # self.main_btn.clicked.connect(self.main)
        # main_layout.addWidget(self.main_btn)
        
        
        self.setLayout(main_layout)
        self.update_image()
    
    def load_example(self):
        # print("Yo")
        # self.image_label.addImage("/Users/clark/Desktop/protein_visualization_app/testing/butterfly.png")
        combined_image = (np.ones(shape=(1000,1000)) * 255).astype("uint8")
        combined_image = np.array(Image.open("/Users/clark/Desktop/protein_visualization_app/testing/butterfly.png"))
        # import PIL
        # PIL.Image.fromarray(combined_image).saveas("hello.png")
        height, width, _ = combined_image.shape
        # bytes_per_line = 2

        q_image = QImage(combined_image.tobytes(), width, height, QImage.Format.Format_RGBA8888) # interesting image.tobytes() works well, maybe you don't need to do bytes_per_line for conversion into qimage anymore,
        # RGB888 also works 
        self.changePix.emit(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))
        
    def addImage(self, pixmap):
        '''add a new image'''

        self.channels = None
        self.resetTransform()
        
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = (pixmap)
        
    
    def add_layer_controls(self, idx):
        print("add alyers controls", idx)
        group_box = QGroupBox(f"Layer {idx + 1}: {self.layer_names[idx]}")
        group_layout = QFormLayout()
        
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        opacity_slider.valueChanged.connect(lambda value: self.update_opacity(value, idx))
        self.opacity_sliders.append(opacity_slider)
        group_layout.addRow("Opacity:", opacity_slider)
        
        contrast_slider = QSlider(Qt.Orientation.Horizontal)
        contrast_slider.setMinimum(50)
        contrast_slider.setMaximum(150)
        contrast_slider.setValue(100)
        contrast_slider.valueChanged.connect(lambda value: self.update_contrast(value, idx))
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
        
        group_box.setLayout(group_layout)
        self.scroll_layout.addWidget(group_box)
    
    def show_layer_dialog(self):
        dialog = LayerDialog(self.layers, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_index = dialog.get_selected_layer_index()
            if selected_index is not None:
                self.add_layer(selected_index)
    
    def show_color_dialog(self, idx):
        print(idx)
        dialog = ColorDialog(self.color_dict, self)
        if dialog.exec_() == QDialog.Accepted:
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
        self.ims.append(new_img)
        self.layer_names.append(new_name)
        self.current_opacities.append(1.0)
        self.current_contrasts.append(1.0)
        self.current_visibilities.append(True)
        self.current_tints.append(QColor(255, 255, 255))
        self.add_layer_controls(len(self.ims) - 1)
        self.update_image()
    
    def update_opacity(self, value, idx):
        self.current_opacities[idx] = value / 100.0
        self.update_image()
    
    def update_contrast(self, value, idx):
        self.current_contrasts[idx] = value / 100.0
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
    
    def adjust_contrast(self, img, min=5, max = 98):
            # pixvals = np.array(img)
            minval = np.percentile(img, min) # room for experimentation 
            maxval = np.percentile(img, max) # room for experimentation 
            img = np.clip(img, minval, maxval)
            img = ((img - minval) / (maxval - minval)) * 255
            return (img.astype(np.uint8))
        
    def update_image(self):
        combined_image = np.zeros_like(self.ims[0], dtype=np.float32)
        
        for img, opacity, contrast, visible, tint in zip(self.ims, self.current_opacities, self.current_contrasts, self.current_visibilities, self.current_tints):
            if visible:
                print(contrast)
                adjusted_img = img * contrast
                adjusted_img = np.clip(adjusted_img, 0, 255)  # Clip values to
                adjusted_img = self.apply_tint(adjusted_img, tint)
                adjusted_img = np.clip(adjusted_img, 0, 255)  # Clip values to stay in the valid range
                combined_image += adjusted_img * opacity
        
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        
        height, width, _ = combined_image.shape
        bytes_per_line = 3

        q_image = QImage(combined_image.tobytes(), width, height, QImage.Format.Format_RGB888) # interesting image.tobytes() works well, maybe you don't need to do 
        
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

