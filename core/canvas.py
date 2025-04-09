from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap,  QCursor, QImage
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
import tifffile as tiff, numpy as np, matplotlib as mpl, time, cv2, xml.etree.ElementTree as ET, os
from core.Worker import Worker
from utils import *
from ui.Dialogs import ImageDialog

class ImageWrapper:
    def __init__(self, data, name="", cmap="gray"):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        
        self.name = name
        self.cmap = cmap
        self.data = data
        self.contrast_min = 0
        self.contrast_max = 255

class __BaseGraphicsView(QWidget):
    '''base class for graphics view'''
    channelLoaded = pyqtSignal(dict, bool)
    channelNotLoaded = pyqtSignal(np.ndarray)
    updateProgress = pyqtSignal(int, str)
    errorSignal = pyqtSignal(str)
    fill_metadata = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        
        self.setMinimumSize(QSize(300, 300))
        self.scene = QGraphicsView()
        self.scene.setScene(QGraphicsScene(self))
        self.reset_pixmap =  None
        self.reset_pixmapItem = None
        self.pixmap = None
        self.pixmapItem=None
        self.np_channels = {}
        self.reset_np_channels = {}
        self.currentChannelNum = -1
        self.is_layered = False

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        event.acceptProposedAction()

    def read_tiff_pages(self, file_path):
        pages = []
        with tiff.TiffFile(file_path) as tif:
            for i, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    # Check if the image is blank
                    if np.all(image == image.flat[0]):
                        continue
                    pages.append(image)
                    self.updateProgress.emit(int((i+1)/len(tif.pages)*100), "Loading image")
                except Exception as e:
                    print(f"Error reading page {i}: {e}")
                    continue

            if pages:
                print("stacked")
                return np.stack(pages)
        
            else:
                return np.stack([])
    

    def filename_to_image(self, file_name:str, adjust_contrast=False) -> np.ndarray:  


            if file_name.endswith((".tiff", ".tif")):


                metadata_widget = MetaData()
                metadata = metadata_widget.parse_metadata(file_name)
                # metadata_widget.populate_table(metadata)
                self.fill_metadata.emit(metadata)
                pages = self.read_tiff_pages(file_name)
                num_channels = pages.shape[0]

                if (num_channels > 1):
                    self.is_layered = True
                    for channel_num, image in enumerate(pages):

                        channel_name = f'Channel {channel_num + 1}'
                        height, width = image.shape
                        image_adjusted = image

                        if adjust_contrast:
                            __scaled = scale_adjust(image) 
                            image_adjusted = adjustContrast(__scaled) # uint8

                        # bytesPerPixel = 2 if image_adjusted.dtype == np.uint16 else 1
                        # format = QImage.Format.Format_Grayscale16 if image_adjusted.dtype == np.uint16 else QImage.Format.Format_Grayscale8

                        self.np_channels[channel_name] = ImageWrapper(image_adjusted) 
                        print('data type: ', type(image_adjusted))
                        self.reset_np_channels[channel_name] = ImageWrapper(image_adjusted.copy()) 



                    channel_one_image = next(iter(self.np_channels.values())).data
                    self.channelLoaded.emit(self.np_channels, True)
                else: # num of channels is 1, single page
                    print("not multilayer")

                    self.is_layered = False
                    self.np_channels.clear()
                    self.reset_np_channels.clear()
                    channel_one_image = pages[0]
                    self.image = channel_one_image
                    self.channelNotLoaded.emit(channel_one_image)
                    self.currentChannelNum = -1
            else: # not a .tif image
                self.is_layered = False
                from PIL import Image
                print("not a tif")
                self.np_channels.clear()
                self.reset_np_channels.clear()
                channel_one_image = np.array(Image.open(file_name))
                self.image = channel_one_image
                self.currentChannelNum = -1

            self.updateProgress.emit(100, "Image Loaded")
            return channel_one_image
    
    def deleteImage(self):
        self.scene.scene().clear()



class ReferenceGraphicsView(__BaseGraphicsView):

    update_reference = pyqtSignal(QPixmap, bool)
    # referenceLoaded = pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_layered = False

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path:
                    self.addImage(file_path)
            event.acceptProposedAction()
    

    def addImage(self, file_path:str):

        self.reference_worker = Worker(self.filename_to_image, file_path, True)
        self.reference_worker.start()
        self.reference_worker.signal.connect(self.filename_to_image_complete)
        self.reference_worker.finished.connect(self.reference_worker.quit)
        self.reference_worker.finished.connect(self.reference_worker.deleteLater)

    def filename_to_image_complete(self, image):
        if not self.np_channels.get("Channel 1") == None:
            self.is_layered = True

        else:
            self.is_layered = False

        qimage = numpy_to_qimage(image)
        self.pixmap = QPixmap(qimage)

        self.update_reference.emit(self.pixmap, self.is_layered)

##########################################################
class ImageGraphicsView(__BaseGraphicsView):
    
    canvasUpdated = pyqtSignal(QPixmap)
    newImageAdded = pyqtSignal(QGraphicsPixmapItem)
    saveImage = pyqtSignal(QGraphicsPixmapItem)  
    changeSlider = pyqtSignal(tuple)

    def __init__(self, parent=None):

        super().__init__(parent)
        self.reset_pixmap=None
        self.reset_pixmapItem=None
        self.pixmap=None
        self.pixmapItem=None
        self.begin_crop = False
        self.crop_cursor =  QCursor(Qt.CursorShape.CrossCursor)

    def toPixmapItem(self, data:QPixmap|np.ndarray|QImage):
        #convert pixmap to pixmapItem
        if type(data) == QPixmap:
            self.pixmap = data
        elif type(data) == QImage:
            self.pixmap = QPixmap(data)
        else:
            self.pixmap = QPixmap(numpy_to_qimage(data))
    
        self.canvasUpdated.emit(self.pixmap)
    
    update_cmap = pyqtSignal(str)
    def change_cmap(self, cmap_text: str):
        '''updates the current image using the current colormap and contrast settings'''
        lut = self.generate_lut(cmap_text)
        adjusted_uint8 = scale_adjust(self.image)
        if self.is_layered:
            channel_num = f"Channel {self.currentChannelNum + 1}"
            self.np_channels[channel_num].cmap = cmap_text
            min, max = self.np_channels[channel_num].contrast_min, self.np_channels[channel_num].contrast_max
            self.update_contrast((min, max))
            print("contrast updated")
        

        rgb = self.label2rgb(adjusted_uint8, lut).astype(np.uint8)
        self.toPixmapItem(rgb)

        self.update_cmap.emit(cmap_text)

  

    def generate_lut(self, cmap:str):
        '''generate a 8 bit look-up table and converts to rgb space'''
        label_range = np.linspace(0, 1, 256)
        return np.uint8(mpl.colormaps[cmap](label_range)[:,2::-1]*256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        '''applys the look-up table and merges r, g, b channels to form colored image '''
        if len(labels) == 3:
            r,g,b = labels
            return cv2.LUT(cv2.merge((r, g, b)), lut)
        else:
            return cv2.LUT(cv2.merge((labels, labels, labels)), lut) # gray to color
    

    def loadStardistLabels(self, stardist: ImageWrapper):
        self.stardist_labels = stardist.data
        self.image = self.stardist_labels.copy()
        self.change_cmap("gray")
    
    def addImage(self, file:str):
        '''add a new image'''
        self.np_channels.clear()
        self.scene.resetTransform()

        if isinstance(file, str):

           self.image_worker = Worker(self.filename_to_image, file)
           self.image_worker.signal.connect(self.onFileNameToPixmapCompleted)
           self.image_worker.error.connect(self.onError)
           self.image_worker.finished.connect(self.image_worker.quit)
           self.image_worker.start()


    @pyqtSlot(object)
    def onFileNameToPixmapCompleted(self, image):
        '''handles operation after the file is loaded into the canvas'''
        if image.dtype != np.uint8:
            image = scale_adjust(image)

        qimage = numpy_to_qimage(image)
        pixmap = QPixmap(qimage)
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.newImageAdded.emit(self.pixmapItem) # emit uint16, change to uint8 in canvas_ui

    
    def pixmap_to_numpy_array(self):
        # Convert the QPixmap to a QImage
        image = self.pixmap.toImage()

        # Get image dimensions
        width = image.width()
        height = image.height()

        # Convert QImage to byte array
        ptr = image.bits()
        ptr.setsize(height * width * 4)

        # Convert byte array to numpy array
        array = np.array(ptr).reshape((height, width, 4))

        return array

    def reset_image(self):
        '''resets the image to original state'''
        if self.pixmapItem: 

            import copy
            self.np_channels = copy.deepcopy(self.reset_np_channels)
            self.channelLoaded.emit(self.np_channels, False)

            channel_num = f"Channel {self.currentChannelNum + 1}"
            self.image = self.np_channels.get(channel_num).data

            layered_data = self.reset_np_channels.get(channel_num).data
            # self.reset_pixmap = QPixmap(numpy_to_qimage(layered_data))
            # self.toPixmapItem(self.reset_pixmap)
            self.change_cmap("gray")

    def rotate_image_task(self, channels:dict, angle):
        t = time.time()
        rotated_arrays = []

        if self.is_layered:
            for wrapper in channels.values():
                try:
                    arr = wrapper.data
                    cmap = wrapper.cmap
                    if not arr.data.contiguous:
                        arr = np.ascontiguousarray(arr, dtype='uint16')
                except Exception as e:
                    print("error: ", str(e)) # should be a QMessageBox

                # rotate image
                h,w = arr.shape
                center = (w/2, h/2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
                cos = np.abs(rotation_matrix[0,0])
                sin = np.abs(rotation_matrix[0,1])
                updated_w = int((h*sin) + (w*cos))
                updated_h = int((h*cos) + (w*sin))
                rotation_matrix[0,2] += (updated_w/2) - w/2
                rotation_matrix[1,2] += (updated_h/2) - h/2
                rotated_arr = cv2.warpAffine(arr, rotation_matrix, (updated_h, updated_h))

                self.rotated_wrapper = ImageWrapper(rotated_arr, cmap = cmap)
                rotated_arrays.append(self.rotated_wrapper)
            return  dict(zip(channels.keys(), rotated_arrays))
        else:
            h,w = self.image.shape
            center = (w/2, h/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
            cos = np.abs(rotation_matrix[0,0])
            sin = np.abs(rotation_matrix[0,1])
            updated_w = int((h*sin) + (w*cos))
            updated_h = int((h*cos) + (w*sin))
            rotation_matrix[0,2] += (updated_w/2) - w/2
            rotation_matrix[1,2] += (updated_h/2) - h/2
            rotated_arr = cv2.warpAffine(self.image, rotation_matrix, (updated_h, updated_h))

            return rotated_arr

    def rotateImage(self, angle_text: str):
        try:
            angle = float(angle_text)
        except ValueError:
            print("Error: Please enter a valid number.") # this should be a QMessageBox
            return

        if self.pixmap and angle is not None:
            self.rotation_worker = Worker(self.rotate_image_task, self.np_channels, angle)
            self.rotation_worker.signal.connect(self.onRotationCompleted) # result is rotated_channels
            self.rotation_worker.error.connect(self.onError)
            self.rotation_worker.finished.connect(self.rotation_worker.quit)
            self.rotation_worker.finished.connect(self.rotation_worker.deleteLater)
            self.rotation_worker.start()

    @pyqtSlot(object)
    def onRotationCompleted(self, result):

        if type(result) == dict:

            self.np_channels = result
            channel_image = list(self.np_channels.values())[self.currentChannelNum].data
            channel_cmap = list(self.np_channels.values())[self.currentChannelNum].cmap
            channel_image = scale_adjust(channel_image)
            self.image = channel_image
            self.change_cmap(channel_cmap) # this also updates the contrast
            self.channelLoaded.emit(self.np_channels, False)

        else:
            channel_image = scale_adjust(result)
            self.image = channel_image
            self.change_cmap("gray") # this also updates the contrast



    @pyqtSlot(str)
    def onError(self, error_message):
        print(f"Error: {error_message}")

    def updateChannels(self, channels:dict, clear:bool) -> None: #cropsignal will update this
        self.np_channels = channels # replace channels with new, cropped/rotated, etc
        self.channelLoaded.emit(self.np_channels, clear)


    def updateCurrentImage(self, data_dict):
        self.image = data_dict[f"Channel {self.currentChannelNum + 1}"].data


    def swapChannel(self, index):
        '''swaps between channels of a multi-layered tiff image'''
        channel_num = f'Channel {index+1}'
        self.currentChannelNum = index

        self.image = self.np_channels.get(channel_num).data # self.image always needs to be updated. this is the current image that is being operated on
        
        if channel_num in self.np_channels.keys():
            self.change_cmap(self.np_channels.get(channel_num).cmap) # this also updates the contrast


    def update_contrast(self, values):

        if self.pixmap is None:
            self.errorSignal.emit("Canvas empty")
            return
        
        min_val, max_val = values
        if self.is_layered:
            channel_num = f"Channel {self.currentChannelNum + 1}"

            self.np_channels.get(channel_num).contrast_min = min_val
            self.np_channels.get(channel_num).contrast_max = max_val


        contrast_image = self.apply_contrast(min_val, max_val)
        
        contrastPixmap = QPixmap(numpy_to_qimage(contrast_image))

        self.canvasUpdated.emit(contrastPixmap)

        self.changeSlider.emit((min_val, max_val))


    def auto_contrast(self, lower = 0.1, upper=.9):
        if self.is_layered:
            channel_num = f"Channel {self.currentChannelNum + 1}"
            channel = scale_adjust(self.np_channels[channel_num].data)

        flat_channel = channel.flatten()
    
        hist, _ = np.histogram(flat_channel, bins=256, range=(0, 255))
        total_pixels = flat_channel.size
        cumulative_hist = np.cumsum(hist) / total_pixels
        new_min= np.argmax(cumulative_hist > lower)  
        new_max = np.argmax(cumulative_hist > upper)  

        self.update_contrast((new_min, new_max))

    def apply_contrast(self, new_min, new_max):

        qimage = self.pixmap.toImage() # get current image
        image = qimage_to_numpy(qimage) # returns uint8
        lut = self.create_lut(new_min, new_max)
        return cv2.LUT(image, lut)

    def create_lut(self, new_min, new_max):

        lut = np.zeros(256, dtype=np.uint8) # uint8 for display
        lut[new_min:new_max+1] = np.linspace(start=0, stop=255, num=(new_max - new_min + 1), endpoint=True, dtype=np.uint8)
        lut[:new_min] = 0 # clip between 0 and 255
        lut[new_max+1:] = 255

        return lut

    def blur_layer(self, blur_percentage:float, confirm=False):
        """
        Applies Gaussian blur chosen of the image stack and subtracts
        the specified percentage of the blurred image from the original.
        """
        self._blur_layer = f'Channel {self.currentChannelNum+ 1}'
        if not confirm:
            
            # blur_percentage = self._blur_percentage
            layer_to_blur = self.np_channels[self._blur_layer].data

            blurred_mask = cv2.GaussianBlur(layer_to_blur, (101, 101), 0)
            blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(np.uint16)
            self.corrected_layer = cv2.subtract(layer_to_blur, blurred_mask_adjusted)
            self.corrected_layer = np.clip(self.corrected_layer, 0, 65535).astype(np.uint16)
            cmap = self.np_channels[self._blur_layer].cmap
            self.image = self.corrected_layer
            self.change_cmap(cmap)

        else:
            print("Error from gaussian blur")


        if confirm and hasattr(self, "corrected_layer") and (not self.np_channels.get(self._blur_layer) == None):
            self.np_channels[self._blur_layer].data = self.corrected_layer # Replace with the corrected version
            self.channelLoaded.emit(self.np_channels, False)
            self.updateProgress.emit(100, f"Replaced {self._blur_layer}")
            

    def showCroppedImage(self, image_rect):
        """Show dialog with cropped image preview"""
        pixmap = self.pixmap 
        cropped = pixmap.copy(image_rect).toImage()
        cropped_pixmap = QPixmap(cropped)

        if not self.is_layered:
            self.image = qimage_to_numpy(cropped)

        
        self.crop_dialog = ImageDialog(self, cropped_pixmap)
        self.crop_dialog.exec()

        if self.crop_dialog.confirm_crop:
            self.crop_worker = Worker(self.cropImageTask, image_rect)
            self.crop_worker.signal.connect(self.onCropCompleted) 
            self.crop_worker.finished.connect(self.crop_worker.quit)
            self.crop_worker.finished.connect(self.crop_worker.deleteLater)
            self.crop_worker.start()
        else:
            self.cropSignal.emit(False)

    cropSignal = pyqtSignal(bool)
    @pyqtSlot(dict)
    def onCropCompleted(self, cropped_wrappers: dict):
        """Handle completed crop operation"""

        if not cropped_wrappers == {}:
            self.np_channels = cropped_wrappers
            channel_num = f"Channel {self.currentChannelNum + 1}"
            self.image = self.np_channels.get(channel_num).data

        self.cropSignal.emit(False)
        print("crop signal emitted")

    def cropImageTask(self, image_rect) -> dict:
        """Process crop in background thread"""
        left = image_rect.x()
        top = image_rect.y()
        right = image_rect.right()
        bottom = image_rect.bottom()


        if self.is_layered:
            for channel_name, image_arr in self.np_channels.items():
                arr = image_arr.data 
                cmap = image_arr.cmap
                cropped_array = arr[top:bottom+1, left:right+1]
                if not cropped_array.data.contiguous:
                    cropped_array = np.ascontiguousarray(cropped_array, dtype="uint16")

                if not hasattr(self, "cropped_wrappers"):
                    self.cropped_wrappers = {}

                self.crop_wrapper = ImageWrapper(cropped_array, cmap=cmap)
                min, max = self.np_channels[channel_name].contrast_min, self.np_channels[channel_name].contrast_max
                self.crop_wrapper.contrast_min = min
                self.crop_wrapper.contrast_max = max
                self.update_contrast((min, max))
                self.cropped_wrappers[channel_name] = self.crop_wrapper

            return self.cropped_wrappers
            
        else: return {}
    

class MetaData(QWidget):
    '''Class to handle metadata of images'''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.row_count = 8
        self.column_count = 1
        self.table = QTableWidget()
        self.table.setRowCount(8)
        self.table.setColumnCount(1)

        self.headers = ['Name', 'URI', 'Pixel Type', 'Width', 
                                 'Height', 'Dimension (CZT)', 'PhysicalSizeX', 'PhysicalSizeY']

        self.table.setVerticalHeaderLabels(self.headers)

        self.table.setHorizontalHeaderLabels(['Value'])
        self.table.setColumnWidth(0, 300)

        editable_fields = [3, 7, 8]
        # for row in range(self.row_count):
            
            # if row in editable_fields:
            #     value_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # Make it read-only

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def set_item(self, row, value):
        item = QTableWidgetItem(value)
        self.table.setItem(row, 0, item)

    def populate_table(self, metadata: dict):
        self.table.setRowCount(len(self.headers))

        for row, key in enumerate(self.headers):
            value = metadata.get(key, "Unknown")  
            value_item = QTableWidgetItem(str(value))

            # key_item.setFlags(key_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, value_item)

    def parse_metadata(self, filename):
        file_name = os.path.basename(filename)  
        name = os.path.splitext(file_name)[0]    

        with tiff.TiffFile(filename) as tif:
            raw_meta_data = {}
            page = tif.pages[0]
            for tag in page.tags.values():
                raw_meta_data[tag.name] = tag.value
                
        try:
            desc = raw_meta_data["ImageDescription"]
            metadata = {}
            root = ET.fromstring(desc)
            namespace_uri = root.tag[root.tag.find('{')+1 : root.tag.find('}')]
            ns = {'ome': namespace_uri}
            pixels = root.find('.//ome:Pixels', namespaces=ns)

            if pixels is not None:

                metadata = {
                    'Name': name,
                    'URI': filename,
                    'Width': pixels.attrib.get('SizeX'),
                    'Height': pixels.attrib.get('SizeY'),
                    'Dimension (CZT)': f"{pixels.attrib.get('SizeC')} x {pixels.attrib.get('SizeZ')} x {pixels.attrib.get('SizeT')}",
                    'Pixel Type': pixels.attrib.get('Type'),
                    'PhysicalSizeX': f"{pixels.attrib.get('PhysicalSizeX')} {pixels.attrib.get('PhysicalSizeXUnit')}",
                    'PhysicalSizeY': f"{pixels.attrib.get('PhysicalSizeY')} {pixels.attrib.get('PhysicalSizeYUnit')}",
                    'DimensionOrder': pixels.attrib.get('DimensionOrder')
                }

                for k, v in metadata.items():
                    print(f"{k}: {v}")
            else:
                print("Pixels element not found.")
                
        except ET.ParseError as e:
            print("Parse error has occurred:", e)

        finally:

            if not metadata:
                metadata["Name"] = name
                metadata["URI"] = filename
                metadata["Width"] = raw_meta_data["ImageWidth"]
                metadata["Height"] = raw_meta_data["ImageLength"]
                metadata['Pixel Type'] = f'uint{raw_meta_data["BitsPerSample"]}'
                metadata['Dimension (CZT)'] = "Unknown"
                metadata['PhysicalSizeX'] = "Unknown"
                metadata['PhysicalSizeY'] = "Unknown"
                metadata['DimensionOrder'] = "Unknown"
    
        return metadata
