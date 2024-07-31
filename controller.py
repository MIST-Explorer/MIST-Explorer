
import ui.app, Dialogs, image_processing.canvas, image_processing.stardist, image_processing.cell_intensity, image_processing.register
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap
import numpy as np
import cv2
from PIL import Image
        
class Controller:
    def __init__(self, 
                 model_canvas: image_processing.canvas.ImageGraphicsView, 
                 model_stardist: image_processing.stardist.StarDist,
                 model_cellIntensity: image_processing.cell_intensity.CellIntensity,
                 model_register: image_processing.register.Register,
                 view: ui.app.Ui_MainWindow):
        
        self.view = view
        self.model_canvas = model_canvas 
        self.model_stardist = model_stardist
        self.model_cellIntensity = model_cellIntensity
        self.model_register = model_register
        self.openFilesDialog = None

        #menubar signals
        # self.view.menubar.actionOpenFiles.triggered.connect(self.on_action_openFiles_triggered)
        self.view.menubar.actionOpenReference.triggered.connect(self.on_action_reference_triggered)
        self.view.menubar.actionOpen.triggered.connect(self.on_actionOpen_triggered)
        
        # self.view.connect()

        #toolbar signals
        self.view.toolBar.actionReset.triggered.connect(self.model_canvas.resetImage)
        self.view.toolBar.actionOpenBrightnessContrast.triggered.connect(self.createBCDialog)
        self.view.toolBar.channelSelector.currentIndexChanged.connect(self.on_channelSelector_currentIndexChanged)



        self.view.canvas.imageDropped.connect(self.model_canvas.addImage)
        self.model_canvas.newImageAdded.connect(self.view.canvas.addNewImage) # loading a new image
        self.view.view_tab.changePix.connect(self.view.canvas.addNewImage) # loading a new image


        self.model_canvas.canvasUpdated.connect(self.view.canvas.updateCanvas) # operation done on current image
        self.model_canvas.channelLoaded.connect(self.view.toolBar.updateChannelSelector)
        self.model_canvas.channelLoaded.connect(self.view.stardist_groupbox.updateChannelSelector)
        self.model_canvas.channelLoaded.connect(self.model_stardist.updateChannels)
        self.model_canvas.channelLoaded.connect(self.view.canvas.loadChannels)
        self.model_canvas.channelNotLoaded.connect(self.view.toolBar.clearChannelSelector)
        self.model_canvas.channelNotLoaded.connect(self.view.stardist_groupbox.clearChannelSelector)
        self.model_canvas.channelNotLoaded.connect(self.model_stardist.setImageToProcess)
        self.model_canvas.updateProgress.connect(self.view.updateProgressBar)

        
        # crop signals
        self.view.crop_groupbox.crop_button.triggered.connect(self.view.canvas.startCrop)
        self.view.crop_groupbox.cancel_crop_button.triggered.connect(self.view.canvas.endCrop)
        
        # confirm rotate signal

        self.view.saveSignal.connect(self.controlSave)

        self.view.rotate_groupbox.rotate_confirm.pressed.connect(lambda: self.model_canvas.rotateImage(self.view.rotate_groupbox.rotate_line_edit.text()))


        #stardist signals
        #change params
        self.view.stardist_groupbox.stardist_channel_selector.currentTextChanged.connect(self.model_stardist.setChannel)
        self.view.stardist_groupbox.stardist_pretrained_models.currentTextChanged.connect(self.model_stardist.setModel)
        self.view.stardist_groupbox.percentile_high.valueChanged.connect(self.model_stardist.setPercentileHigh)
        self.view.stardist_groupbox.percentile_low.valueChanged.connect(self.model_stardist.setPercentileLow)
        self.view.stardist_groupbox.prob_threshold.valueChanged.connect(self.model_stardist.setProbThresh)
        self.view.stardist_groupbox.nms_threshold.valueChanged.connect(self.model_stardist.setNMSThresh)
        self.view.stardist_groupbox.n_tiles.valueChanged.connect(self.model_stardist.setNumberTiles)

        self.view.stardist_groupbox.radius.valueChanged.connect(self.model_stardist.setDilationRadius)


        #run stardist
        self.view.stardist_groupbox.stardist_run_button.pressed.connect(self.model_stardist.runStarDist)

        # display stardist result
        self.model_stardist.stardistDone.connect(self.model_canvas.toPixmapItem)
        self.model_stardist.sendGrayScale.connect(self.model_cellIntensity.loadStardistLabels)
        # self.model_stardist.stardistDone.connect(self.view.view_tab.loadStarDistLabels)

        # generate cell data signals
        # change params
        self.view.cellIntensity_groupbox.alignment_layer.currentTextChanged.connect(self.model_register.setAlignmentLayer)
        self.view.cellIntensity_groupbox.protein_cell_layer.currentTextChanged.connect(self.model_register.setCellLayer)
        self.view.cellIntensity_groupbox.intensity_layer.currentTextChanged.connect(self.model_register.setProteinDetectionLayer)
        self.view.cellIntensity_groupbox.overlap.valueChanged.connect(self.model_register.setOverlap)
        self.view.cellIntensity_groupbox.max_size.valueChanged.connect(self.model_register.setMaxSize)
        self.view.cellIntensity_groupbox.num_cycles.valueChanged.connect(self.model_cellIntensity.setNumDecodingCycles)
        self.view.cellIntensity_groupbox.num_layers_each.valueChanged.connect(self.model_cellIntensity.setNumDecodingColors)
        self.view.cellIntensity_groupbox.num_tiles.valueChanged.connect(self.model_cellIntensity.setNumTiles)
        self.view.cellIntensity_groupbox.radius_fg.valueChanged.connect(self.model_cellIntensity.setRadiusFG)
        self.view.cellIntensity_groupbox.radius_bg.valueChanged.connect(self.model_cellIntensity.setRadiusBG)

        self.view.cellIntensity_groupbox.bead_data.pressed.connect(self.model_cellIntensity.loadBeadData)
        self.view.cellIntensity_groupbox.bead_data.pressed.connect(self.model_cellIntensity.loadColorCode)
        self.view.cellIntensity_groupbox.run_button.pressed.connect(self.model_cellIntensity.run)
        

        # Display Butterfly
        self.model_stardist.stardistDone.connect(self.model_canvas.toPixmapItem)
        
        # Save photo
        self.model_stardist.stardistDone.connect(self.model_canvas.toPixmapItem)
        
        # tab switched
        self.view.tabWidget.currentChanged.connect(lambda x: self.view.small_view.setVisible(not bool(x)))


    def on_action_openFiles_triggered(self):
        self.openFilesDialog = Dialogs.OpenFilesDialog(self.view)
        self.openFilesDialog.show()

    def save_pixmap_as_image(self, pixmap: QPixmap, filename: str):
        
        
        # Convert QPixmap to QImage
        qimage = pixmap.toImage()

        # Convert QImage to numpy array
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

        # Save numpy array as an image file using OpenCV
        cv2.imwrite(filename, cv2.cvtColor(arr, cv2.COLOR_BGRA2BGRA))
        
    def pixmap_to_image(self, pixmap: QPixmap):
        
        if pixmap == None:
            return None
        # Convert QPixmap to QImage
        qimage = pixmap.toImage()

        # Convert QImage to numpy array
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

        # Save numpy array as an image file using OpenCV
        return arr
    
    def is_grayscale(image: np.ndarray) -> bool:

        if len(image.shape) == 3 and image.shape[2] == 3:
            return False
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            return True
        else:
            raise ValueError("Image format not recognized")
        
    def controlSave(self):

        pm = self.model_canvas.pixmap
        print(pm)
        if pm != None:
            im = self.pixmap_to_image(pm)
            
        
            
            
            file_name, _ = QFileDialog.getSaveFileName(None, "Save File", "image.png","All Files(*);;Text Files(*.txt)")
            if file_name:
                print(file_name)
                Image.fromarray(im).save(file_name)
                
            else:
                return False
    def createBCDialog(self):
        self.BC_dialog = Dialogs.BrightnessContrastDialog(self, self.model_canvas.channels, self.view.canvas, self.view.toolBar.operatorComboBox)

    def openFileDialog(self, viewer):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        if file_name:
            viewer.addImage(file_name)

    def on_action_reference_triggered(self):
       self.openFileDialog(self.view.small_view)

    def on_actionOpen_triggered(self):
       self.openFileDialog(self.model_canvas)  

    def on_channelSelector_currentIndexChanged(self, index):
        if self.model_canvas.channels:
            channel_pixmap = QPixmap.fromImage(self.model_canvas.channels[self.view.toolBar.channelSelector.itemText(index)])
            self.model_canvas.toPixmapItem(channel_pixmap)
