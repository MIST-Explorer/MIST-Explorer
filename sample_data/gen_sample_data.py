import numpy as np, tifffile as tiff
import pandas as pd

def crop_tiff(image_path, x, y):
    image_data = tiff.imread(image_path) 
    
    image_data = image_data[:, 0:y, 0:x]
    
    tiff.imwrite('cropped_output.tif', image_data)
    
def crop_csv(file_path, threshold):
    df = pd.read_csv(file_path)
    filtered_df = df[(df.iloc[:, 1] < threshold) & (df.iloc[:, 2] < threshold)]
    filtered_df.to_csv("cropped_cell_data.csv", index=False)

    
# crop_tiff("/Users/clark/Downloads/protein signal.ome.tif", 4000, 4000)
crop_csv("/Users/clark/Desktop/protein_visualization_app/sample_data/celldta.csv", 4000)
# def crop_data_gen(image_path, orig_cell_data_path):
    
