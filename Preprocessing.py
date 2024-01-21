import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
import glob
import os,time
import random
from PIL import Image
import shutil
import pydicom

Cancer_dicom = sorted(glob.glob('./VinDr/images/train/*.png'))

# function to return cropped dicom image using cv2.connectedComponentsWithStats
def crop_dicom(dicom_file):
    result = cv2.connectedComponentsWithStats((dicom_file > 0.005).astype(np.uint8)[:, :], 8, cv2.CV_32S)
    # results 4 outputs ref: https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
    # But for reconizing the black color(0) from others(1), we just need result[2] , called stat matrix
    stat_matrix = result[2]  # include left, top, width, height, area_size columns
    # the second row shows the boxes of pixels which are different from background.Also, we don't need are_size cloumn. So:
    second_row = stat_matrix[1:, 4].argmax() + 1
    x1, y1, w, h = stat_matrix[second_row][:4]
    x2 = x1 + w
    y2 = y1 + h
    cropped_dicom = dicom_file[y1: y2, x1: x2]
    return cropped_dicom

# fucntion to read dicom image and return it's pixel
def read_dicom(dicom_file):
    dicom = pydicom.dcmread(dicom_file)  # read dciom files
    dicom_pixel = dicom.pixel_array  # read the pixel of images

    # Standardize with transferig to [0,1] space
    dicom_pixel = (dicom_pixel - dicom_pixel.min()) / (dicom_pixel.max() - dicom_pixel.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        #invert the pixel value
        dicom_pixel = 1 - dicom_pixel

    return dicom_pixel

def dicom_to_png(file, save_folder="", extension='png'):
    image_id = file.split(".")[0]
    image_id = image_id.split('/')[7]
    image_id = str(image_id)
    new_size = (1024,1280)
    dicom_pixel = read_dicom(dicom_file=file)  # read dicom image pixels
    # dicom_pixel = pydicom.pixel_data_handlers.util.apply_modality_lut(dicom_data)
    # dicom_pixel = pydicom.pixel_data_handlers.util.apply_voi_lut(dicom_data)
    '''cropped_dicom = crop_dicom(dicom_file=dicom_pixel)  # crop the dicom image'''
    resized_img = cv2.resize(dicom_pixel, dsize = new_size)  # resize it to specific size

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # Apply CLAHE to the image
    clahe_image = clahe.apply(resized_img)

    cv2.imwrite(save_folder + f"{clahe_image}.{extension}", (resized_img * 255).astype(np.uint8))

    msg = print(image_id," converted from dicom to png successfully")
    return msg

files = Cancer_dicom[:1286]

for file in files:
    dicom_to_png(file, save_folder="/content/drive/MyDrive/VinDr_N/images/train/", extension='png')

fools = Cancer_dicom[1286:1606]
for fool in fools:
    dicom_to_png(fool, save_folder="/content/drive/MyDrive/VinDr_png/images/val/", extension='png')

# Read the CSV file containing ROI coordinates and labels
csv_file = pd.read_csv('/content/drive/MyDrive/VinDr_Dicom/vindr.csv')
yolo_annotations = []

# Convert ROI coordinates to YOLO format
def yolo_format(image_width, image_height, xmin, ymin, xmax, ymax):
    x_center = (xmin + ((xmax - xmin) / 2)) / image_width
    y_center = (ymin + ((ymax - ymin) / 2)) / image_height
    norm_width = (xmax - xmin) / image_width
    norm_height = (ymax - ymin) / image_height
    return x_center, y_center, norm_width, norm_height

# Create the directory if it doesn't exist

annotations_dir = '/content/sample_data/YOLO/'
os.makedirs(annotations_dir, exist_ok=True)

# Iterate over each row in the CSV file and generate YOLO format annotations
for i, row in csv_file.iterrows():
    image_id = row['image_id']
    view = row['view_position']
    laterality = row['laterality']
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    image_width = row['width']
    image_height = row['height']

    image_label = f'{image_id}_{laterality}_{view}'

    x_center, y_center, norm_width, norm_height = yolo_format(image_width, image_height, xmin, ymin, xmax, ymax)
    yolo_annotation = f"0 {x_center} {y_center} {norm_width} {norm_height}"
    yolo_annotations.append(yolo_annotation)

    # Save the YOLO annotations to a text file
    annotation_path = os.path.join(annotations_dir, f'{image_label}.txt')
    with open(annotation_path, 'a') as file:  # Use 'a' to append instead of 'w' to overwrite
        file.write(yolo_annotation + '\n')  # Add a new line after each annotation
    print(f"{i} YOLO annotations", image_label, "saved successfully")
