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
from PyQt6.QtCore import pyqtSignal
import re
import tqdm

class Register:
    # registrationDone = pyqtSignal(np.ndarray)
    def __init__(self):
        Image.MAX_IMAGE_PIXELS = 99999999999    
        self.protein_signal_array = None
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
                "path": r"C:\\Users\\jianx\Downloads\\test\\cycle_1.ome.tif", 
                # "flor_layers": [3, 4, 5], # this actually doesnot do anything in this program
                "brightfield": 0, 
                "pystack_transforms" : [],
                "sitk_transforms": []
            },
            {   
                "path": r"C:\\Users\\jianx\Downloads\\test\\protein signal.ome.tif", 
                # "flor_layers": [2, 3],   # this actually does not do anything in this program
                "brightfield": 0, #alignment layer
                "pystack_transforms" : [],
                "sitk_transforms": [] 
            },
        )

    def runRegistration(self):
        m = self.params["max_size"]
        self.OVERLAP = self.params["overlap"]
        self.NUM_TILES = self.params['num_tiles']
        basis = self.tifs[0]
        print("opening files!")
        bf1_f = Image.open(basis["path"])
        bf1_f.seek(basis["brightfield"])
        bf1 = np.array(bf1_f)
        bf1 = bf1[0:m, 0:m]
        bf1 = self.adjust_contrast(bf1,50, 99)
        fixed_map = TileMap("fixed", bf1, self.OVERLAP, self.NUM_TILES)

        # generate tiles
        for tif_n, tif in enumerate(self.tifs):
            
            bf2_f = Image.open(self.tifs[tif_n]["path"])
            # bf2_f.seek(tifs[tif_n]["brightfield"])
            bf2 = np.array(bf2_f)
            
            bf2 = bf2[0:m, 0:m]
            bf2 = self.adjust_contrast(bf2,50, 99)

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

            # 

            
            # Select the inputs number
            outputs = []
            for tile_n, tile_set in enumerate(inputs):
                try:
                    print(f"aligned a tile...{tile_n}")
                    if (tif_n == 0):
                        outputs.append(self.onskip(tile_set))
                        continue
                        
                    t = time.time()
                    result = self.f(tile_set)
                    print(time.time() - t)
                    if result == None:
                        continue
                    outputs.append(result)
                except Exception as e:
                    raise e
                    print("ERROE!")
                    print(e)
            print("done aligning")

            self.tifs[tif_n]["outputs"] = outputs
 
#########################################################
        # align all layers
        aligned_protein_signal = None
        import tifffile

        test_mode = True

        for i, tif in enumerate(self.tifs): 
            # if test_mode and i == 0:
            #     continue
            
            print(tif["path"])
            file = Image.open(tif["path"])
            new_registered_tif = []
            
            for layer_number in range(file.n_frames):

                # # testing!
                # if layer_number == 0:
                #     continue

                print("Layer Number:", layer_number, "for tif", i)
                
                file.seek(layer_number) 
                bf1_f.seek(layer_number)
                
                bf = np.array(file)
                bf1 = np.array(bf1_f)

                if bf.shape[0] < m:
                    raise Exception("too small! only", bf.shape[0], m)
                    continue

                # bf = sc(bf)
                # bf1 = sc(bf1)
                
                bf = bf[0:m, 0:m]
                # bf = adjust_contrast(bf, 50, 99)
                
                dest = Image.fromarray(np.zeros((m, m), dtype="float"))  #need to determine the final bit size
                
                for result in tif["outputs"]:
                    im, x, y, metadata = result
                    transforms, ymin, xmin, radius, _, _ = metadata

                    corresponding_tile = None
                    
                    if transforms == None:
                        # print("transforms is none")
                        corresponding_tile = moving_map.get_tile_by_center(bf, x, y)[ymin: ymin + radius * 2, xmin: xmin + radius * 2]
            
                    else:
                        print(i)
                        im, x, y, metadata = result
                        transforms, ymin, xmin, radius, _, _ = metadata
                        
                        transf = transforms[0]
                        x_translate, y_translate =  transforms[1]
                        sr =  transforms[2]
                        
                        source = moving_map.get_tile_by_center(bf, x, y).astype(float)
                        target = moving_map.get_tile_by_center(bf1, x, y).astype(float)

                        # registered = source
                        
                        
                        registered, footprint = aa.apply_transform(transf, source, target)
                        
                        print("source", source.max())
                        print(registered.max())
                        
                        registered = sr.transform(registered)
                    
                        corresponding_tile = registered[ymin: ymin + radius * 2, xmin: xmin + radius * 2]

                
                        # corresponding_tile = cv2.copyMakeBorder(corresponding_tile, 0,1,0,1, cv2.BORDER_REPLICATE) 
                        print(corresponding_tile.max())

                    dest.paste(Image.fromarray(pystackreg.util.to_uint16(corresponding_tile)), (int(x - radius), int(y - radius)))
            
                dest_arr = np.array(dest)
                new_registered_tif.append(dest_arr)
                # print(dest_arr.shape)
                # testing!
            #     dest.convert("L").save(f"layer{i}registered.png")

            #     print(f"saved layer{i}registered.png")

            #     break

            # continue

            new_registered_tif = [x.astype("uint16") for x in new_registered_tif]
            new_registered_tif = np.stack(new_registered_tif)

            print(new_registered_tif.shape)
            aligned_protein_signal = new_registered_tif
            # skimage.io.imsave(f"C:\\Users\\Administrator\\Desktop\\Clark Fischer's Files\\test_{i}.tif", new_registered_tif)

        self.protein_signal_array = aligned_protein_signal[self.params['protein_detection_layer'], :, :][0:self.params['max_size'], 0:self.params['max_size']] # -> use to generate cell intensity table
        # cell_image = aligned_protein_signal[protein_signal_cell_layer, :, :][0:max_size, 0:max_size] # -> stardist
        # self.registrationDone.emit(protein_signal_array)
        # self.registrationDone.emit(cell_image)

    def setAlignmentLayer(self, channel):
        match = re.search(r'\d+', channel)
        if match:
            number = int(match.group()) 
            result = number - 1  # 0 index
            print(result) 
            self.params['alignment_layer'] = result

    def setCellLayer(self, channel):
        match = re.search(r'\d+', channel)
        if match:
            number = int(match.group()) 
            result = number - 1  # 0 index
            print(result) 
        self.params['cell_layer'] = result

    def setProteinDetectionLayer(self, channel):
        match = re.search(r'\d+', channel)
        if match:
            number = int(match.group()) 
            result = number - 1  # 0 index
            print(result) 
        self.params['cell_layer'] = result
        self.params['protein_detection_layer'] = result

    def setMaxSize(self, value):
        self.params['max_size'] = value

    def setNumTiles(self, value):
        self.params['num_tiles'] = value

    def setOverlap(self, value):
        self.params['overlap'] = value

    def run(self):
        pass


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

    def f(self, param):
        fixed_img, moving_img, ymin, xmin, radius, x, y = param
        #### insert custom code here
        
        source = moving_img.copy()
        target = fixed_img.copy()

        source = np.clip(source,128, 255 )-128  # to remove most gradient of background
        target = np.clip(target, 128, 255)-128  # to remove most gradient of background
        
        # def convert_to_rgb_image(array_2d):
        #     # Normalize the input array to be between 0 and 1
        #     normalized_array = (array_2d - np.min(array_2d)) / (np.max(array_2d) - np.min(array_2d))
            
        #     # Rescale the values to be between 0 and 255
        #     scaled_array = (normalized_array * 255).astype(np.uint8)
            
        #     # Create a 3D array where each channel is the same as the input array
        #     rgb_image = np.stack((scaled_array, scaled_array, scaled_array), axis=-1)
            
        #     return rgb_image
        
        # Convert 2D array to RGB image
        # source = convert_to_rgb_image(source)
        # target = convert_to_rgb_image(target)
        print(source.dtype, source.shape, target.dtype, target.shape)
        transf, (source_list, target_list) = aa.find_transform(source, target,detection_sigma=2, min_area=20, max_control_points=500)
        registered, footprint = aa.apply_transform(transf, source, target)
    
        # # convert the 3D back to 2D gray image
        # def rgb2gray(rgb):
        #     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        # registered = rgb2gray(registered)
        # target = rgb2gray(target)

        sr = StackReg(StackReg.AFFINE)
        sr.register(target, registered)
        # registered = sr.transform(registered)

        print()
        
        return (
            (None, x, y, ([transf, (0, 0), sr], ymin, xmin, radius, x, y))
        )

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

    def remove_large_blobs(self, image):
        out = inputs[24][0] - np.array(dip.AreaOpening(inputs[24][0], filterSize=150, connectivity=2))
        return np.array(out)


    def equalize_shape(self, cy1_rescale, cy2_rescale):
        [cy1x, cy1y] = cy1_rescale.shape
        [cy2x, cy2y] = cy2_rescale.shape
        
        pos = lambda x: 0 if x < 0 else x
        
        # print(pos(cy1x-cy2x), pos(cy1y-cy2y))
        cy2_rescale = np.pad(cy2_rescale, ((int(math.floor(pos(cy1x-cy2x)/2)),int(math.ceil(pos(cy1x-cy2x)/2))),(math.floor((pos(cy1y-cy2y)/2)),math.ceil((pos(cy1y-cy2y)/2)))), 'empty')
        # Sometimes "edge" might work better
            
        cy2_rescale = cy2_rescale[0:cy1x, 0:cy1y]
        
        return cy1_rescale, cy2_rescale
    

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
