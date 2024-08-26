
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, QObject, QThread
from PyQt6.QtWidgets import QMessageBox, QFileDialog
import numpy as np, cv2 as cv, math, time, pandas as pd, itertools
from image_processing.canvas import ImageGraphicsView
from image_processing.register import Register
import ui.app
from PIL import Image
from tqdm import tqdm

# import SimpleITK as sitk

class CellIntensity:
    def __init__(self):


        self.params = {
        'alignment_layer': 'Channel 1',
        'cell_layer': 'Channel 3',
        'protein_detection_layer' : 'Channel 4',
        'max_size': 23000,
        'num_tiles': 5,
        'overlap': 500,
        'num_decoding_cycles': 3,
        'num_decoding_colors': 3,
        'radius_fg': 2,
        'radius_bg': 6

        
        }
        self.bead_data = pd.read_csv("sample_data/bead_data.csv").to_numpy().astype("uint16")
        print("bead data init")
        self.color_code = pd.read_csv("sample_data/ColorCode.csv")
        self.stardist_labels = np.array(Image.open("testing/dilated_stardist_labels.tif"))

      
    def generateCellIntensityTable(self):
        # registration code
        self.reg_thread = QThread()
        self.reg_task = Register()
        self.reg_task.moveToThread(self.reg_thread)
        self.reg_thread.start()
        self.reg_thread.started.connect(self.reg_task.runRegister)
        self.protein_signal_array = self.reg_task.protein_signal_array


        if (self.bead_data.any()==None):
            QMessageBox.critical(ui.app.Ui_MainWindow(), "Error", "Please select all necessary parameters")

        else:


            possible_value_of_layers = list(range(0, self.params['num_decoding_colors']))
            all_protein_permutations = [''.join([str(x) for x in p]) for p in itertools.product(possible_value_of_layers, repeat=self.params['num_decoding_cycles'])]
            color_code_to_index = {k: i for i, k in enumerate(all_protein_permutations)}

            # then convert string to int (for faster access/comparison in numpy)
            color_code_to_index = {int(key): value for key, value in color_code_to_index.items()}

            # and the inverse to go from array index -> protein
            index_to_color_code = {value: int(key) for key, value in color_code_to_index.items()}



            cell_data_dict = {}
            num_proteins = len(color_code_to_index)
            if self.stardist_labels.any():
                for cell_id in range(1, np.max(self.stardist_labels) + 1):
                    cell_data_dict[cell_id] = []
                    for i in range(num_proteins):
                        cell_data_dict[cell_id].append([])

            else:
                print("stardist not done yet")

            data_modified = np.zeros((len(self.bead_data), 3))
            data_modified[:, 0:2] = self.bead_data[:, 0:2].astype("uint16")
            data_modified[:, 2] = np.array([''.join(str(x) for x in bead[2:2+self.params['num_decoding_cycles']]) for bead in self.bead_data])
            data_modified = data_modified
            

            radius_bg = self.params['radius_bg']
            radius_fg = self.params['radius_fg']
            radius = radius_bg - radius_fg

            # this code (with radius = 2, radius_bg = 6)...
            bool_arr = np.zeros((1+radius_bg*2, 1+radius_bg*2)) # fill with zeros
            bool_arr[:, :radius] = 1 # left wall
            bool_arr[:radius, :] = 1 # top wall
            bool_arr[:, (1+radius_bg*2) - radius:] = 1 # right wall 
            bool_arr[(1+radius_bg*2) - radius:, :] = 1 # bottom wall

            # ...creates this structure
            # bool_arr = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 1
            #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 2
            #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 3
            #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 4
            #             [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], # 5 
            #             [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], # 6
            #             [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], # 7
            #             [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], #
            #             [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], #
            #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #
            #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #
            #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ]

            # then this is the inner ball and the outer loops
            self.__background_bool_arr = np.array(bool_arr).astype("bool")
            self.__bead_bool_arr = np.invert(self.__background_bool_arr)


            for bead in tqdm(data_modified):
                bead_x, bead_y = bead[0:2]
                bead_x, bead_y = int(bead_x), int(bead_y)
                cell_associated_id = self.stardist_labels[bead_y, bead_x]
                
                # 0 is the background :) 
                if cell_associated_id != 0:
                    color_code = bead[2]
                    # bounds checking
                    max_size = self.params['max_size']
                    if bead_x > radius_bg and bead_y > radius_bg and bead_x < (max_size - radius_bg) and bead_y < (max_size - radius_bg):
                    # if True:
                        # see function above
                        adjusted_median_intensity = self.get_adjusted_median_intensity(bead_x, bead_y)
                        cell_data_dict[cell_associated_id][color_code_to_index[color_code]].append(adjusted_median_intensity)


            beads_split_by_protein = {}
            for i, key in enumerate(color_code_to_index):
                beads_split_by_protein[i] = data_modified[data_modified[:, 2] == key][:, 0:2]

            cell_centroids = {}
            for cell_id in tqdm(cell_data_dict):
                cell_centroids[cell_id] = self.find_centerpoint_of_cell(cell_id)



            print("Finding values for cells with incomplete protein profiles")
            for cell_id in tqdm(cell_data_dict):
                # the number of beads for any given protein for any cell 
                num_beads_each_protein_each_cell = [len(x) for x in cell_data_dict[cell_id]]

                if 0 in num_beads_each_protein_each_cell:
                    # if theres a 0 in that array, it means some protein DOES NOT have a bead for that protein
                    # in that case, we take this conditional and we need to find the nearest bead with that protein
                    proteins_index = [i if val == 0 else -1 for i, val in enumerate(num_beads_each_protein_each_cell)] 

                    # this distills it into the proteins which are not accounted for (by index)
                    missing_proteins = list(filter(lambda x: x != -1, proteins_index))

                    # get cell centroid for step below
                    cell_center_x, cell_center_y = cell_centroids[cell_id]
                    # then we search for the nearest neighbor, in the pre-filtered array (see above)
                    # this pre-filtered array contains only beads of the same protein as is missing.
                    # we of course repeat this for every protein that is missing.
                    for missing_protein_index in missing_proteins:
                        # find the nearest neighbor to centroid (within the filtered list)
                        cell_protein_nn_x, cell_protein_nn_y = self.find_nearest_neighbor([cell_center_x, cell_center_y], beads_split_by_protein[missing_protein_index])
                        cell_protein_nn_x, cell_protein_nn_y = int(cell_protein_nn_x), int(cell_protein_nn_y)
                        # store in cell dict, as normal
                        if cell_protein_nn_x > radius_bg and cell_protein_nn_y > radius_bg and cell_protein_nn_x < (max_size - radius_bg) and cell_protein_nn_y < (max_size - radius_bg):
                            adjusted_median_intensity = self.get_adjusted_median_intensity(cell_protein_nn_x, cell_protein_nn_y)
                            cell_data_dict[cell_id][missing_protein_index].append(adjusted_median_intensity)



            median_values_for_cell_data_dict = {}

            # set up as before for `cell_data_dict`
            for cell_id in cell_data_dict:
                median_values_for_cell_data_dict[cell_id] = []
                for i in range(num_proteins):
                    median_values_for_cell_data_dict[cell_id].append([])

            # populate with medians
            for cell_id in cell_data_dict:
                array_of_subarrays = cell_data_dict[cell_id]
                array_of_subarrays_medians = [np.median(subarr) for subarr in array_of_subarrays]
                
                median_values_for_cell_data_dict[cell_id] = array_of_subarrays_medians




            # drop rows with NaN that pandas includes for some reason lol
            self.color_code = self.color_code.dropna(how='all', axis=1).dropna(how='all', axis=0)
            self.color_code = self.color_code.to_numpy()

            # lets us go from code -> protein i.e. 112 -> Fox3 or whatever
            color_code_translation_dict = {}
            for row in self.color_code:
                protein_name = row[0] 
                code = (int("".join([str(int(x)) for x in row[1:]])))
                color_code_translation_dict[code] = protein_name


            # then we use this to build the header string
            header = ["Global X", "Global Y"]
            for subarray_index in index_to_color_code:
                coresponding_protein_code = index_to_color_code[subarray_index]
                # print(coresponding_protein_code)
                if coresponding_protein_code in color_code_translation_dict:
                    readable__protein_name = color_code_translation_dict[coresponding_protein_code]
                    header.append(readable__protein_name)
                else:
                    header.append("N/A")

            # Now get all the data out of the subarrays
            save_this = np.array([v for k,v in median_values_for_cell_data_dict.items()])
            # and all the centroid data
            save_this = np.hstack(([v for k,v in cell_centroids.items()], save_this))
            # and finally save everything
            df_cell_data = pd.DataFrame(save_this, columns=header) #--> use this to visualize

    def find_nearest_neighbor(query_point, data_points):
        # Calculate the Euclidean distance from the query point to each data point
        distances = np.linalg.norm(data_points - query_point, axis=1)
        # Find the index of the nearest neighbor
        nearest_index = np.argmin(distances)
        # Return the nearest neighbor point
        return data_points[nearest_index]


    def find_centerpoint_of_cell(self, cell_id):
        # get just the cell with the label given
        just_cell_id_stardist_labels = (self.stardist_labels==cell_id).astype("uint8") 
        # standard centroid finding with cv.findContours
        contours, _ = cv.findContours(just_cell_id_stardist_labels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])  
        
        return (cx, cy)


    def get_adjusted_median_intensity(self, bead_x, bead_y):
        
        radius_bg = self.params['radius_bg']
        # radius_fg = self.params['radius_fg']
        # get entire 13x13 region (or whatever size)
        region = self.protein_signal_array[bead_y-radius_bg:bead_y+radius_bg+1, bead_x-radius_bg:bead_x+radius_bg+1]
        # use boolean indexing to only get outside ring (donut)
        bg_pixels = region[self.__background_bool_arr]
        # use boolean indexing to only get inside (donut hole)
        bead_pixels = region[self.__bead_bool_arr]

        # find 80th percentile 
        percentile_bg = np.percentile(bg_pixels, 80)
        # adjust bead intensity relative to
        return np.median(bg_pixels)- percentile_bg*0.3

    def loadStardistLabels(self, stardist_labels_grayscale):
        self.stardist_labels = stardist_labels_grayscale
    
    def loadBeadData(self):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Bead Data", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        if file_name:
            self.bead_data = pd.read_csv(file_name).to_numpy().astype("uint16")  # this is the output from the registration->decoding program

    def loadColorCode(self):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Color Code", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        if file_name:
            self.color_code = pd.read_csv(file_name)


    def setAlignmentLayer(self, channel):
        self.params['alignment_layer'] = channel

    def setCellLayer(self, model):
        self.params['cell_layer'] = model

    def setProteinDetectionLayer(self, value):
        self.params['protein_detection_layer'] = value

    def setMaxSize(self, value):
        self.params['max_size'] = value

    def setNumTiles(self, value):
        self.params['num_tiles'] = value

    def setOverlap(self, value):
        self.params['overlap'] = value

    def setNumDecodingCycles(self, value):
        self.params['num_decoding_cycles'] = value

    def setNumDecodingColors(self, value):
        self.params['num_decoding_colors'] = value

    def setRadiusFG(self, value):
        self.params['radius_fg'] = value

    def setRadiusBG(self, value):
        self.params['radius_bg'] = value

