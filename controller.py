import ui.app, Dialogs, image_processing.canvas, image_processing.stardist
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap
import numpy as np
import cv2
        
class Controller:
    def __init__(self, 
                 model_canvas: image_processing.canvas.ImageGraphicsView, 
                 model_stardist: image_processing.stardist.StarDist,
                 view: ui.app.Ui_MainWindow):
        
        self.view = view
        self.model = model_canvas 
        self.model_stardist = model_stardist
        self.openFilesDialog = None

        #menubar signals
        self.view.menubar.actionOpenFiles.triggered.connect(self.on_action_openFiles_triggered)
        self.view.menubar.actionOpenReference.triggered.connect(self.on_action_reference_triggered)
        self.view.menubar.actionOpen.triggered.connect(self.on_actionOpen_triggered)
        
        # self.view.connect()

        #toolbar signals
        self.view.toolBar.actionReset.triggered.connect(self.model.resetImage)
        self.view.toolBar.actionOpenBrightnessContrast.triggered.connect(self.createBCDialog)
        self.view.toolBar.channelSelector.currentIndexChanged.connect(self.on_channelSelector_currentIndexChanged)


        self.view.canvas.imageDropped.connect(self.model.addImage)
        self.model.newImageAdded.connect(self.view.canvas.addNewImage) # loading a new image
        self.view.view_tab.changePix.connect(self.view.canvas.addNewImage) # loading a new image
        # self.view.view_tab.getPix.connect(self.view.canvas.addNewImage) # loading a new image

        self.model.canvasUpdated.connect(self.view.canvas.updateCanvas) # operation done on current image
        self.model.channelLoaded.connect(self.view.toolBar.updateChannelSelector)
        self.model.channelLoaded.connect(self.view.stardist_groupbox.updateChannelSelector)
        self.model.channelLoaded.connect(self.model_stardist.updateChannels)
        self.model.channelNotLoaded.connect(self.view.toolBar.clearChannelSelector)
        self.model.channelNotLoaded.connect(self.view.stardist_groupbox.clearChannelSelector)
        self.model.channelNotLoaded.connect(self.model_stardist.setImageToProcess)

        
        # crop signals
        self.view.crop_groupbox.crop_button.triggered.connect(self.view.canvas.startCrop)
        self.view.crop_groupbox.cancel_crop_button.triggered.connect(self.view.canvas.endCrop)
        
        # confirm rotate signal
        self.view.rotate_groupbox.rotate_confirm.pressed.connect(lambda: self.model.rotateImage(self.view.rotate_groupbox.rotate_line_edit.text()))
        
        #
        self.view.saveSignal.connect(self.controlSave)

        #stardist signals
        #change params
        self.view.stardist_groupbox.stardist_channel_selector.currentTextChanged.connect(self.model_stardist.setChannel)
        self.view.stardist_groupbox.stardist_pretrained_models.currentTextChanged.connect(self.model_stardist.setModel)
        self.view.stardist_groupbox.percentile_high.valueChanged.connect(self.model_stardist.setPercentileHigh)
        self.view.stardist_groupbox.percentile_low.valueChanged.connect(self.model_stardist.setPercentileLow)
        self.view.stardist_groupbox.prob_threshold.valueChanged.connect(self.model_stardist.setProbThresh)
        self.view.stardist_groupbox.nms_threshold.valueChanged.connect(self.model_stardist.setNMSThresh)
        self.view.stardist_groupbox.n_tiles.valueChanged.connect(self.model_stardist.setNumberTiles)
        self.view.stardist_groupbox.kernel_size.valueChanged.connect(self.model_stardist.setDilationKernelSize)
        self.view.stardist_groupbox.iterations.valueChanged.connect(self.model_stardist.setDilationIterations)


        #run stardist
        self.view.stardist_groupbox.stardist_run_button.pressed.connect(self.model_stardist.runStarDist)

        # display stardist result
        self.model_stardist.stardistDone.connect(self.model.toPixmapItem)
        
        # Display Butterfly
        self.model_stardist.stardistDone.connect(self.model.toPixmapItem)
        
        # Save photo
        self.model_stardist.stardistDone.connect(self.model.toPixmapItem)

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
        
        pm = self.model.pixmap
        print(pm)
        if pm != None:
            im = self.pixmap_to_image(pm)
            
        else:
            print("nothin loaded!")
            
        print("controling saving")

    def createBCDialog(self):
        self.BC_dialog = Dialogs.BrightnessContrastDialog(self, self.model.channels, self.view.canvas, self.view.toolBar.operatorComboBox)

    def openFileDialog(self, viewer):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        if file_name:
            viewer.addImage(file_name)

    def on_action_reference_triggered(self):
       self.openFileDialog(self.view.small_view)

    def on_actionOpen_triggered(self):
       self.openFileDialog(self.model)  

    def on_channelSelector_currentIndexChanged(self, index):
        if self.model.channels:
            channel_pixmap = QPixmap.fromImage(self.model.channels[self.view.toolBar.channelSelector.itemText(index)])
            self.model.toPixmapItem(channel_pixmap)

    # def update_param(self, key, value):
    #     self.model_stardist.set_param(key, value)