import numpy as np
import cv2
import math
import diplib as dip
import SimpleITK as sitk
import statistics
import sep
import astroalign as aa
from pystackreg import StackReg
from PIL import Image
import time
import pystackreg.util
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot, QObject
import re
import tqdm

class Register(QThread):
    cell_image_signal = pyqtSignal(np.ndarray)
    protein_signal_arr_signal = pyqtSignal(np.ndarray)  
    imageReady = pyqtSignal(bool)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = 99999999999  

        # initialize variables

        self.protein_channels = None 
        self.reference_channels = None
        self.protein_signal_array = None
        self.has_blue = True  
        self.params = { 
            'alignment_layer': 0, 
            'cell_layer': 2, # 0 index
            'protein_detection_layer': 3, # 0 index
            'max_size': 23000,
            'num_tiles': 5,
            'overlap': 500,
        }
        self.tifs = (
            { 
                "image_dict": self.reference_channels,
            },

            {
                "image_dict": self.protein_channels,
            },
        )
    def run_registration(self):
        
        # run on gpu if possible
        import tensorflow as tf
        gpu = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu:
            device_name= tf.test.gpu_device_name()
            print("gpu name: ", device_name)
        else:
            device_name = '/CPU:0'

        with tf.device(device_name):
            self.start()
    
            self.finished.connect(self.quit)
            self.finished.connect(self.deleteLater)

    def run(self):
        

        m = self.params["max_size"]
        self.OVERLAP = self.params["overlap"]
        self.NUM_TILES = self.params['num_tiles']
        basis = self.tifs[0]
        print("opening files!")
        self.progress.emit(0, "preparing alignment")
        print(basis["image_dict"] is None)
        bf1_f = basis["image_dict"]
        bf1 = basis["image_dict"][f"Channel {self.params['alignment_layer']  + 1}"].data # 0 index so add 1 to avoid key error

        bf1 = self.adjust_contrast(bf1,50, 99)
        bf1 = bf1[0:m, 0:m]

        fixed_map = TileMap("fixed", bf1, self.OVERLAP, self.NUM_TILES)

        # generate tiles
        for tif_n, tif in enumerate(self.tifs):

            # skip reference
            if tif_n == self.params["alignment_layer"]:
                self.tifs[tif_n]["outputs"] = None
                continue
            
            bf2_f = self.tifs[tif_n]["image_dict"][f"Channel {self.params['alignment_layer'] + 1}"].data

            bf2 = self.adjust_contrast(bf2_f, 50, 99)
            bf2 = bf2[0:m, 0:m]

            print(bf1.shape, bf2.shape)

            moving_map = TileMap("moving", bf2, self.OVERLAP, self.NUM_TILES)

            inputs = []
            radius = int(fixed_map.tile_size)
            for mov_data, fix_data in list(zip(moving_map, fixed_map)):
                
                (moving_img, moving_bounds) = mov_data
                (fixed_img, fixed_bounds) = fix_data
                
                x, y = moving_bounds["center"]
                ymin = moving_bounds["ymin"]
                xmin = moving_bounds["xmin"]
            
                radius = int(fixed_map.tile_size)
            
                inputs.append((fixed_img, moving_img, ymin, xmin, radius, x, y))

            
            # Select the inputs number
            outputs = []
            for tile_n, tile_set in enumerate(inputs):
                try:

                    # update progress bar
                    print(f"aligned a tile...{tile_n}")
                    progress_update = int(((tile_n+1)/len(inputs))*100)
                    self.progress.emit(progress_update, str(f"aligning tile {tile_n+1}/{len(inputs)}"))

                    print(1)
                    if (tif_n == 0):
                        outputs.append(self.onskip(tile_set))
                        continue
                    
                    print("2")
                    t = time.time()
                    result = self.align_two_img(tile_set) # align
                    print(time.time() - t)
                    if result == None:
                        continue
                    outputs.append(result)
                except Exception as e:
                    print("ERROR!")
                    raise e
                    
            print("done aligning")

            self.tifs[tif_n]["outputs"] = outputs
            
 
#########################################################
        # align all layers
        aligned_protein_signal = None

        for i, tif in enumerate(self.tifs): 

            if i == 0:
                continue
            
            # print(tif["path"])
            # file = Image.open(tif["path"])
            file = tif["image_dict"]
            n_frames = len(file) # 4
            print("n frames", n_frames)
            new_registered_tif = []
            
            for layer_number in range(n_frames):    

                print("Layer Number:", layer_number, "for tif", i)
                progress_update = int(((layer_number+1)/n_frames)*100)
                self.progress.emit(progress_update, f"Layer Number: {layer_number+1} for tif {i+1}")
                
                bf = file[f'Channel {layer_number + 1}'].data# channels are index 1
                bf1 = bf1_f[f'Channel {layer_number + 1}'].data  #channels are index 1 # this is the basis
             

                if bf.shape[0] < m:
                    raise Exception("too small! only", bf.shape[0], m) # should be a QMessageBox error

                bf = bf[0:m, 0:m]
                
                dest = Image.fromarray(np.zeros((m, m), dtype="float"))  #need to determine the final bit size
                
                for result in tif["outputs"]:
                    transforms, ymin, xmin, radius, x, y = result

                    corresponding_tile = None
                    
                    if transforms == None:
                        corresponding_tile = moving_map.get_tile_by_center(bf, x, y)[ymin: ymin + radius * 2, xmin: xmin + radius * 2]
            
                    else:
                        transforms, ymin, xmin, radius, x, y = result
                        print(x, y)

                        transf = transforms[0]
                        x_translate, y_translate =  transforms[1]
                        
                        source = moving_map.get_tile_by_center(bf, x, y).astype(float)
                        target = moving_map.get_tile_by_center(bf1, x, y).astype(float)

                        
                        
                        registered, footprint = aa.apply_transform(transf, source, target)
                        
                        print("source", source.max())
                        print(registered.max())
                        
                        if self.has_blue:
                            sr =  transforms[2]
                            registered = sr.transform(registered)
                    
                        corresponding_tile = registered[ymin: ymin + radius * 2, xmin: xmin + radius * 2]

                
                        # corresponding_tile = cv2.copyMakeBorder(corresponding_tile, 0,1,0,1, cv2.BORDER_REPLICATE) 
                        print(corresponding_tile.max())

                    dest.paste(Image.fromarray(pystackreg.util.to_uint16(corresponding_tile)), (int(x - radius), int(y - radius)))
            
                dest_arr = np.array(dest)
                new_registered_tif.append(dest_arr)
 

            new_registered_tif = [x.astype("uint16") for x in new_registered_tif]
            new_registered_tif = np.stack(new_registered_tif)

            print(new_registered_tif.shape)
            aligned_protein_signal = new_registered_tif

        self.protein_signal_array = aligned_protein_signal[self.params['protein_detection_layer'], :, :][0:self.params['max_size'], 0:self.params['max_size']] # -> use to generate cell intensity table
        # self.protein_signal_array = aligned_protein_signal[self.params['protein_detection_layer'], :, :]
        cell_image = aligned_protein_signal[self.params['cell_layer'], :, :][0:self.params['max_size'], 0:self.params['max_size']] # -> stardist
        # cell_image = aligned_protein_signal[self.params['cell_layer'], :, :] # --> cell-image
        self.progress.emit(100, "Alignment Done")
        self.protein_signal_arr_signal.emit(self.protein_signal_array) #->cell intensity table
        self.cell_image_signal.emit(cell_image) #-> stardist


    def hasBlueColor(self, hasblue) -> bool:
        if hasblue == "Yes":
            self.has_blue = True
        else: self.has_blue = False
    
    def setAlignmentLayer(self, channel):
        match = re.search(r'\d+', channel)
        if match:
            number = int(match.group()) 
            result = number - 1  # 0 index
            self.params['alignment_layer'] = result
            print("alignment layer is: ", self.params['alignment_layer'])

    def setCellLayer(self, channel):
        match = re.search(r'\d+', channel)
        if match:
            number = int(match.group()) 
            result = number - 1  # 0 index
        self.params['cell_layer'] = result
        print("cell layer is: ", self.params['cell_layer'])

    def setProteinDetectionLayer(self, channel):
        match = re.search(r'\d+', channel)
        if match:
            number = int(match.group()) 
            result = number - 1  # 0 index
        self.params['protein_detection_layer'] = result
        print("protein_detection_layer is: ", self.params['protein_detection_layer'])

    def setMaxSize(self, value):
        self.params['max_size'] = value

    def setNumTiles(self, value):
        self.params['num_tiles'] = value

    def setOverlap(self, value):
        self.params['overlap'] = value

    def onskip(self, param):
        fixed_img, moving_img, ymin, xmin, radius, x, y = param
        return (
                (None, x, y, (None, ymin, xmin, radius, x, y))
            )

    def sc(self, arr):
        return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        # return arr


    def seperate(self, img):
        sep.set_extract_pixstack(10**7)
        
        filename = ''
        threshold = 3
        
        img = img.astype('float')
        
        bkg = sep.Background(img)
        
        img = img - bkg

        return self.sc(img)

    def align_two_img(self, param):

        def convert_to_rgb_image(array_2d):
            normalized_array = (array_2d - np.min(array_2d)) / (np.max(array_2d) - np.min(array_2d))
            scaled_array = (normalized_array * 255).astype(np.uint8)
            rgb_image = np.stack((scaled_array, scaled_array, scaled_array), axis=-1)
            return rgb_image

        fixed_img, moving_img, ymin, xmin, radius, x, y = param        
        source = moving_img.copy()
        target = fixed_img.copy()


        source = np.clip(source,128, 255 )-128  # to remove most gradient of background
        target = np.clip(target, 128, 255)-128  # to remove most gradient of background
        

        print(target.shape, "target dimension")
        print(source.shape, 'source dimension')

        global transf
        global transf_previous

        print("finding alignment")
        try:
            transf, (source_list, target_list) = aa.find_transform(source, target,detection_sigma=3, min_area=10, max_control_points=300)


        except Exception as e:
            print("This tile is not aligned!")
            if 'transf_previous' in globals():
                transf = transf_previous
            else:
                transf = None  # or some default transformation
        finally:
            print(" ")

        transf_previous = transf

        return [transf, []], ymin, xmin, radius, x, y

        # registered, footprint = aa.apply_transform(transf, source, target)
    
        # # # convert the 3D back to 2D gray image
        # # def rgb2gray(rgb):
        # #     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        # # registered = rgb2gray(registered)
        # # target = rgb2gray(target)

        # [transf, []], ymin, xmin, radius, x, y


        # if not self.has_blue:
        #     return (
        #     (None, x, y, ([transf, (0, 0), registered], ymin, xmin, radius, x, y))
        # )

        # else:
        #     sr = StackReg(StackReg.AFFINE)
        #     sr.register(target, registered)
        #     # registered = sr.transform(registered)        
        #     return (
        #         (None, x, y, ([transf, (0, 0), sr], ymin, xmin, radius, x, y))
        #     )
        
        

    def APPLY(self, transformation, fixed, moving):
        registered, footprint = aa.apply_transform(transformation, fixed, moving)
        return registered



    def adjust_contrast(self, img, min=2, max = 98):
                # pixvals = np.array(img)

                minval = np.percentile(img, min) # room for experimentation 
                maxval = np.percentile(img, max) # room for experimentation 
                img = np.clip(img, minval, maxval)
                img = ((img - minval) / (maxval - minval)) * 255
                return (img.astype(np.uint8))

    def resample(self, image, transform):
                # Output image Origin, Spacing, Size, Direction are taken from the reference
                # image in this call to Resample
                reference_image = image
                interpolator = sitk.sitkCosineWindowedSinc
                default_value = 100.0
                return sitk.Resample(image, reference_image, transform,
                                    interpolator, default_value)



    def equalize_shape(self, cy1_rescale, cy2_rescale):
        [cy1x, cy1y] = cy1_rescale.shape
        [cy2x, cy2y] = cy2_rescale.shape
        
        pos = lambda x: 0 if x < 0 else x
        
        # print(pos(cy1x-cy2x), pos(cy1y-cy2y))
        cy2_rescale = np.pad(cy2_rescale, ((int(math.floor(pos(cy1x-cy2x)/2)),int(math.ceil(pos(cy1x-cy2x)/2))),(math.floor((pos(cy1y-cy2y)/2)),math.ceil((pos(cy1y-cy2y)/2)))), 'empty')
        # Sometimes "edge" might work better
            
        cy2_rescale = cy2_rescale[0:cy1x, 0:cy1y]
        
        return cy1_rescale, cy2_rescale
    
    def updateChannels(self, channels) -> None:
        print("debugging", channels)
        self.protein_channels = channels
        self.tifs[1]["image_dict"] = channels
        # print("protein signal images sent to register", self.tifs[0]["image_dict"] is None)
        if not self.reference_channels is None:
            self.imageReady.emit(True)
            print("protein signal image updated")

    def updateCycleImage(self, reference_channels:dict) -> None:
        self.reference_channels = reference_channels
        self.tifs[0]["image_dict"] = reference_channels
        # print("cycle images sent to register", self.tifs[1]["image_dict"] is None)
        if not self.protein_channels is None:
            self.imageReady.emit(True)
            print("cycle image updated")
            
    def cancel(self):

        # self.exit? 
        # self.quit? 
        self.terminate()


############################
class TileMap():
    def __init__(self, name: str, image: np.ndarray, overlap: int, height_width: int):
        """
        :param name:
        :param image:
        :param overlap: pixel amount of overlap
        :param height_width:
        """

        self.name = name
        self.image = image
        
        self.height_width = height_width

        self.tile_center_points = self.blockify(height_width) * self.image.shape[0]

        self.tile_size = self.tile_center_points[0][0][0]

        self.overlap = overlap
        
    
    @staticmethod
    def find_mask(moving_array):
        import diplib as dip
        from PIL import Image
        import numpy as np
        
        def blur(img):
            img = img.copy()
            kernel = np.ones((5,5),np.float)/225
            dst = cv2.filter2D(img,-1,kernel)
            return dst

        def threshold(im, percentile):
            p = np.percentile(im, percentile)
            im = im.copy()
            im[im < p]  = 0
            im[im >= p]  = 255
            return im

        small = cv2.resize(moving_array, (np.array(moving_array.shape) / 10).astype(int), interpolation = cv2.INTER_LINEAR)

        im = np.invert(threshold(blur(small), 20))

        out = dip.AreaOpening(im, filterSize=150, connectivity=2)
        out = np.array(out)

        big = cv2.resize(out, (np.array(moving_array.shape) ).astype(int), interpolation = cv2.INTER_LINEAR)
        big[moving_array == 0] = 255
        
        return np.invert((big / 255).astype(bool))
    

    def get_tile_by_center(self, image, x, y):
        y = round(y)
        x = round(x)
        tile_size = round(self.tile_size) + self.overlap

        return image[self.keep_in_bounds(y - tile_size): self.keep_in_bounds(y + tile_size),
               self.keep_in_bounds(x - tile_size): self.keep_in_bounds(x + tile_size)]
        
    
    def get_bounds_of_tile(self, x, y):
        # print("Got ", x, y)
        tile_size = round(self.tile_size) + self.overlap
        ymin = self.overlap if self.keep_in_bounds(int(y - tile_size)) == int(y - tile_size) else 0
        ymax = self.overlap if self.keep_in_bounds(int(y + tile_size)) == int(y + tile_size) else 0
        xmin = self.overlap if self.keep_in_bounds(int(x - tile_size)) == int(x - tile_size) else 0
        xmax = self.overlap if self.keep_in_bounds(int(x + tile_size)) == int(x + tile_size) else 0

        return {
                    "center": (x,y),
                    "ymin": ymin,
                    "ymax": ymax,
                    "xmin": xmin,
                    "xmax": xmax,
                }

    def __iter__(self):
        for i in self.tile_center_points:
            for j in i:
                # print("THIS IS THE TILE WE TALKIGN ABOUT", j)
                tile = self.get_tile_by_center(self.image, j[0], j[1])
                bounds = self.get_bounds_of_tile(j[0], j[1])
                
                yield (tile, bounds)

    def keep_in_bounds(self, num):
        if num < 0:
            return 0
        if num > self.image.shape[0]:
            return self.image.shape[0]

        return int(num)

    @staticmethod
    def blockify(cuts):
        centerpoints = []
        for i in range(cuts):
            row = []
            for j in range(cuts):
                # print((i + 1), cuts, (j + 1), cuts)
                row.append(np.array([(2 * i + 1) / (cuts * 2), (2 * j + 1) / (cuts * 2)]))
                # print((2*i + 1) / (cuts *2))

            centerpoints.append(np.array(row))

        return np.array(centerpoints)
