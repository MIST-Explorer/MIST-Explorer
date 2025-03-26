
from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtWidgets import QMessageBox, QFileDialog
import numpy as np, cv2 as cv, math, time, pandas as pd, itertools
from core.canvas import ImageGraphicsView, ImageType
from tqdm import tqdm

# import SimpleITK as sitk

class CellIntensity(QThread):
    errorSignal = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    def __init__(self):
        super().__init__()
        self.params = {
        'max_size': 23000,
        'num_decoding_cycles': 3,
        'num_decoding_colors': 3,
        'radius_fg': 2,
        'radius_bg': 6
        }

        self.stardist_labels = None
        self.df_cell_data = None

    def loadProteinSignalArray(self, arr):
        self.protein_signal_array = arr

    def generateCellIntensityTable(self):
        
        self.start()
        self.finished.connect(self.quit)
        self.finished.connect(self.deleteLater)

    def run(self):
        if (self.stardist_labels is None or self.bead_data is None or self.color_code is None):
            self.errorSignal.emit("Please select all necessary parameters")
            return

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
            for cell_id in range(1, np.max(self.stardist_labels) + 1):
                cell_data_dict[cell_id] = []
                for i in range(num_proteins):
                    cell_data_dict[cell_id].append([])


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


            for i, bead in enumerate(data_modified):
                progress_update = int(((i+1)/len(data_modified))*100)
                self.progress.emit(progress_update, f"Adjusting bead intensity {i+1}/{len(data_modified)}")      
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
            for i, cell_id in enumerate(cell_data_dict):
                progress_update = int(((i+1)/len(cell_data_dict))*100)
                self.progress.emit(progress_update, f"Finding center point of cell {i+1}/{len(cell_data_dict)}")
                cell_centroids[cell_id] = self.find_centerpoint_of_cell(cell_id)



            print("Finding values for cells with incomplete protein profiles")
            for i, cell_id in enumerate(cell_data_dict):
                progress_update = int(((i+1)/len(cell_data_dict))*100)
                self.progress.emit(progress_update, f"Finding values for cells with incomplete protein profiles {i+1}/{len(cell_data_dict)}")
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
                try:
                    protein_name = row[0] 
                    code = (int("".join([str(int(x)) for x in row[1:]])))
                    color_code_translation_dict[code] = protein_name
                except ValueError:
                    code = None


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
            self.df_cell_data = pd.DataFrame(save_this, columns=header) #--> use this to visualize
            self.progress.emit(100, "Cell Data is Generated")
            self.saveCellData()

    def cancel(self):
        self.terminate()

        
    def saveCellData(self):
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Cell Data File", "cell_data.csv", "*.csv;;*.xlsx;; All Files(*)")
        if not self.df_cell_data is None:
            self.df_cell_data.to_csv(file_name, index=False)
        else:
            self.errorSignal.emit("Cannot save. Cell data not available")
        

    def find_nearest_neighbor(self, query_point, data_points):
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

        if self.protein_signal_array is None:
            return
        
        radius_bg = self.params['radius_bg']
        radius_fg = self.params['radius_fg']


         # Extract the 5x5 region around the bead
        bead_region = self.protein_signal_array[bead_y-radius_fg:bead_y+radius_fg+1, bead_x-radius_fg:bead_x+radius_fg+1]

        # Calculate the mean and median intensity of the 5x5 bead region
        mean_5x5 = np.mean(bead_region)
        bead_median_org = np.median(bead_region)
        bead_median=bead_median_org.copy()

        # Extract the 15x15 surrounding region

        surrounding_region = self.protein_signal_array[bead_y-radius_bg:bead_y+radius_bg+1, bead_x-radius_bg:bead_x+radius_bg+1]   # Convert to float to handle NaN values

        # Ensure the 15x15 region is valid
        if surrounding_region.shape != (15, 15):
            return bead_median_org  # Return unadjusted median if the 15x15 region is invalid


        # Mask out the 5x5 region from the 15x15 region
        surrounding_region[bead_y-radius_fg:bead_y+radius_fg+1, bead_x-radius_fg:bead_x+radius_fg+1] = 0

        # Calculate the mean intensity of the surrounding 15x15 area, excluding the 5x5 region
        surrounding_mean_15x15 = np.nanmean(surrounding_region)

        # Apply correction only if 15x15 mean is 1.5x greater than 5x5 mean, and bead median > 5000
        if surrounding_mean_15x15 > 1.5 * mean_5x5 and bead_median > 5000:
            # Calculate the correction factor and apply linear correction
            correction_factor = mean_5x5 * (mean_5x5 / surrounding_mean_15x15)
            y =self.linear_correction(correction_factor)

            # Apply the correction to the bead median
            bead_median = bead_median - y + 2000

        # Ensure no negative values
        if bead_median < 1:
            bead_median = 1

        # Return the final adjusted bead median
        return bead_median


    # Define the linear function for the correction equation
    def linear_correction(self, x):
        return 0.8266 * x + 3970.1
    

    def loadStardistLabels(self, stardist:ImageType) ->None:
        self.stardist_labels = stardist.arr
    

    def getBeadData(self, bead_data):
       if isinstance(bead_data, np.ndarray):
          self.bead_data = bead_data

    def getColorCode(self, color_code):
       if isinstance(color_code, pd.DataFrame):
          self.color_code = color_code

    def setNumDecodingCycles(self, value):
        self.params['num_decoding_cycles'] = value

    def setNumDecodingColors(self, value):
        self.params['num_decoding_colors'] = value

    def setRadiusFG(self, value):
        self.params['radius_fg'] = value

    def setRadiusBG(self, value):
        self.params['radius_bg'] = value

