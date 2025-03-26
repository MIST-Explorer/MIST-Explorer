import ui.app, ui.Dialogs as Dialogs, core.canvas, core.stardist, core.cell_intensity, core.register
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
import numpy as np
import cv2
from PyQt6.QtCore import pyqtSignal
from PIL import Image
from ui.Dialogs import ImageDialog
        
class Controller:
    controllerSignal = pyqtSignal(object)
    def __init__(self, 
                 model_canvas: core.canvas.ImageGraphicsView, 
                 model_stardist: core.stardist.StarDist,
                 model_cellIntensity: core.cell_intensity.CellIntensity,
                 model_register: core.register.Register,
                 view: ui.app.Ui_MainWindow):
        
        self.view = view
        self.model_canvas = model_canvas 
        self.model_stardist = model_stardist
        self.model_cellIntensity = model_cellIntensity
        self.model_register = model_register
        self.openFilesDialog = None
        self._small_view = core.canvas.ReferenceGraphicsView()
        #menubar signals
        # self.view.menubar.actionOpenFiles.triggered.connect(self.on_action_openFiles_triggered)
        self.view.menubar.actionOpenReference.triggered.connect(self.on_action_reference_triggered)
        self.view.menubar.actionOpen.triggered.connect(self.on_actionOpen_triggered)
        
        # self.view.connect()
        self.view.register_groupbox.has_blue_color.currentTextChanged.connect(self.model_register.hasBlueColor)
        #toolbar signals
        self.view.toolBar.actionReset.triggered.connect(self.model_canvas.resetImage)
        # self.view.toolBar.actionOpenBrightnessContrast.triggered.connect(self.createBCDialog)
        self.view.toolBar.channelChanged.connect(self.model_canvas.swapChannel)
        # self.view.toolBar.channelChanged.connect(self.view.canvas.setCurrentChannel) #prob should move crop image function to image_processing instead in the future
        # self.view.toolBar.channelChanged.connect(self.model_canvas.setCurrentChannel) # for rotating image
        self.view.toolBar.contrastSlider.valueChanged.connect(self.model_canvas.update_contrast)
        self.view.toolBar.cmapChanged.connect(self.model_canvas.change_cmap) # change cmap in model_canvas then send to view.canvas for display
        # self.model_canvas.timer.timeout.connect(self.model_canvas.update_contrast)
        self.view.toolBar.auto_contrast_button.clicked.connect(self.model_canvas.auto_contrast)
        # self.model_canvas.changeSlider.connect(self.view.toolBar.update_contrast_slider)

        # self.view.canvas.resizeSignal.connect(self.view.reposition)
        self.view.canvas.imageDropped.connect(self.model_canvas.addImage)
        self.view.small_view.imageDropped.connect(self._small_view.addImage)
        self._small_view.referenceViewAdded.connect(self.view.small_view.display)
        self.model_canvas.newImageAdded.connect(self.view.canvas.addNewImage) # loading a new image
        self.view.view_tab.changePix.connect(self.view.canvas.addNewImage) # loading a new image
        self._small_view.channelLoaded.connect(self.model_register.updateCycleImage)
        self._small_view.channelLoaded.connect(self.view.small_view.load_channels)

        self.model_canvas.canvasUpdated.connect(self.view.canvas.updateCanvas) # operation done on current image
        # self.model_canvas.channelLoaded.connect(self.view.toolBar.updateChannels) # update toolbar channels
        self.model_canvas.channelLoaded.connect(self.view.toolBar.updateChannelSelector) # update toolbar channel combobox
        self.model_canvas.channelLoaded.connect(self.view.stardist_groupbox.updateChannelSelector) #update stardist channel combobox
        self.model_canvas.channelLoaded.connect(self.view.register_groupbox.updateChannelSelector) #update stardist channel combobox
        self.model_canvas.channelLoaded.connect(self.view.canvas.loadChannels) #this is for cropping because cropping function is in canvas ui
        self.model_canvas.channelLoaded.connect(self.model_stardist.updateChannels) #pass the channels for stardist processing
        self.model_canvas.channelLoaded.connect(self.model_register.updateChannels)
        self.model_canvas.channelLoaded.connect(self.view.gaussian_blur.updateChannelSelector)
        self.model_canvas.channelNotLoaded.connect(self.view.toolBar.clearChannelSelector) #if new image loaded is not multilayer
        self.model_canvas.channelNotLoaded.connect(self.view.stardist_groupbox.clearChannelSelector)#if new image loaded is not multilayer
        self.model_canvas.channelNotLoaded.connect(self.model_stardist.setImageToProcess) #this is when image loaded does not have multiple layers
        self.model_canvas.updateProgress.connect(self.view.updateProgressBar) # loading image progress bar
        self.model_canvas.errorSignal.connect(self.handleError)
        self.view.canvas.showCrop.connect(self.model_canvas.showCroppedImage)
        self.model_canvas.cropSignal.connect(self.view.canvas.set_crop_status)
        self.model_canvas.update_cmap.connect(self.view.toolBar.update_cmap_selector)
        self.model_canvas.changeSlider.connect(self.view.toolBar.update_contrast_slider)

        # crop signals
        self.view.crop_groupbox.crop_button.triggered.connect(lambda: self.view.canvas.set_crop_status(True)) 
        self.view.crop_groupbox.cancel_crop_button.triggered.connect(lambda: self.view.canvas.set_crop_status(False))
        
        # confirm rotate signal

        self.view.saveSignal.connect(self.controlSave)

        self.view.rotate_groupbox.rotate_confirm.pressed.connect(lambda: self.model_canvas.rotateImage(self.view.rotate_groupbox.rotate_line_edit.text()))

        self.view.gaussian_blur.combo_box.currentIndexChanged.connect(self.model_canvas.set_blur_layer)
        self.view.gaussian_blur.spin_box.valueChanged.connect(self.model_canvas.set_blur_percentage)
        self.view.gaussian_blur.confirm.clicked.connect(self.model_canvas.blur_layer)

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
        self.model_stardist.stardistDone.connect(self.model_canvas.loadStardistLabels) #probably better to use a super class for all model classes so we don't repeat this code
        self.model_stardist.stardistDone.connect(self.model_cellIntensity.loadStardistLabels)
        self.model_stardist.errorSignal.connect(self.handleError)
        self.model_stardist.progress.connect(self.view.updateProgressBar)
        self.view.view_tab.progress.connect(self.view.updateProgressBar)
        self.view.stardist_groupbox.save_button.clicked.connect(self.model_stardist.saveImage)


        # registration
        # change params
        self.view.register_groupbox.alignment_layer.currentTextChanged.connect(self.model_register.setAlignmentLayer)
        self.view.register_groupbox.protein_cell_layer.currentTextChanged.connect(self.model_register.setCellLayer)
        self.view.register_groupbox.intensity_layer.currentTextChanged.connect(self.model_register.setProteinDetectionLayer)
        self.view.register_groupbox.overlap.valueChanged.connect(self.model_register.setOverlap)
        self.view.register_groupbox.max_size.valueChanged.connect(self.model_register.setMaxSize)
        self.view.register_groupbox.num_tiles.valueChanged.connect(self.model_register.setNumTiles)
        self.view.register_groupbox.run_button.clicked.connect(self.model_register.runRegister)
        self.model_register.cell_image_signal.connect(self.model_stardist.loadCellImage)
        self.model_register.protein_signal_arr_signal.connect(self.model_cellIntensity.loadProteinSignalArray)

        # generate cell data
        self.view.cellIntensity_groupbox.bead_data.clicked.connect(self.view.cellIntensity_groupbox.loadBeadData)
        self.view.cellIntensity_groupbox.color_code.clicked.connect(self.view.cellIntensity_groupbox.loadColorCode)
        self.view.cellIntensity_groupbox.emitBeadData.connect(self.model_cellIntensity.getBeadData)
        self.view.cellIntensity_groupbox.emitColorCode.connect(self.model_cellIntensity.getColorCode)
        self.view.cellIntensity_groupbox.num_cycles.valueChanged.connect(self.model_cellIntensity.setNumDecodingCycles)
        self.view.cellIntensity_groupbox.num_layers_each.valueChanged.connect(self.model_cellIntensity.setNumDecodingColors)
        self.view.cellIntensity_groupbox.radius_fg.valueChanged.connect(self.model_cellIntensity.setRadiusFG)
        self.view.cellIntensity_groupbox.radius_bg.valueChanged.connect(self.model_cellIntensity.setRadiusBG)
        self.view.cellIntensity_groupbox.run_button.clicked.connect(self.model_cellIntensity.generateCellIntensityTable)

        self.model_cellIntensity.errorSignal.connect(self.handleError)
        self.view.cellIntensity_groupbox.save_button.clicked.connect(self.model_cellIntensity.saveCellData)        

        self.model_register.progress.connect(self.view.updateProgressBar)
        self.model_cellIntensity.progress.connect(self.view.updateProgressBar)
        
        # tab switched - update small view visibility
        self.view.stackedWidget.currentChanged.connect(lambda x: self.view.small_view.setVisible(not bool(x)))

        # cancel process
        self.view.register_groupbox.cancel_button.clicked.connect(self.model_register.cancel)
        self.view.cellIntensity_groupbox.cancel_button.clicked.connect(self.model_cellIntensity.cancel)
        self.view.stardist_groupbox.cancel_button.clicked.connect(self.model_stardist.cancel)

    def handleError(self, error_message):
        QMessageBox.critical(self.view,"Error", error_message)

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
        # import qimage2ndarray
        # a =qimage2ndarray.recarray_view(qimage)
        # arr = np.array(a)
        # Convert QImage to numpy array
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width*3)
        arr = np.array(ptr).reshape(height, width, 3)  # 4 for RGBA
        # import cv2
        # cv2.imshow("test cropping", arr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
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
        # qimage = pm.toImage()
        if pm != None:
            im = self.pixmap_to_image(pm)
            # import qimage2ndarray 
            # im = qimage2ndarray.alpha_view(qimage)
            file_name, _ = QFileDialog.getSaveFileName(None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)")
            if file_name:
                print(file_name)
                Image.fromarray(im).save(file_name)
                
            else:
                return False
            
        else:
            self.handleError("No image in canvas, please load image")
            

    # def createBCDialog(self):
    #     self.BC_dialog = Dialogs.BrightnessContrastDialog(self, self.model_canvas.channels, self.view.canvas, self.view.toolBar.operatorComboBox)

    def openFileDialog(self, viewer):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.tif);;All Files (*)")
        if file_name:
            viewer.addImage(file_name)

    def on_action_reference_triggered(self):
       self.openFileDialog(self._small_view)

    def on_actionOpen_triggered(self):
       self.openFileDialog(self.model_canvas)


    # def on_channelSelector_currentIndexChanged(self, index):
    #     if self.view.toolBar.channelSelector.count() != 0:
    #         print("current index: ", self.view.toolBar.channelSelector.currentIndex())
    #         # self.controllerSignal.emit(self.view.toolBar.channelSelector.currentIndex())
    #         self.view.canvas.setCurrentChannel(self.view.toolBar.channelSelector.currentIndex()) 
    #         channel_pixmap = QPixmap.fromImage(self.model_canvas.channels[self.view.toolBar.channelSelector.itemText(index)])
    #         self.model_canvas.toPixmapItem(channel_pixmap)


