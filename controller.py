import ui.app, Dialogs, image_processing.canvas, image_processing.stardist
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap

from cf_global import SingletonData

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

        #toolbar signals
        self.view.toolBar.actionReset.triggered.connect(self.model.resetImage)
        self.view.toolBar.actionOpenBrightnessContrast.triggered.connect(self.createBCDialog)
        self.view.toolBar.channelSelector.currentIndexChanged.connect(self.on_channelSelector_currentIndexChanged)


        self.view.canvas.imageDropped.connect(self.model.addImage)
        self.model.newImageAdded.connect(self.view.canvas.addNewImage) # loading a new image
        self.view.view_tab.changePix.connect(self.view.canvas.addNewImage) # loading a new image

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

    def on_action_openFiles_triggered(self):
        self.openFilesDialog = Dialogs.OpenFilesDialog(self.view)
        self.openFilesDialog.show()

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