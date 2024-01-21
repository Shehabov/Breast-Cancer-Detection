# Breast-Cancer-Detection
Yolov8 based breast cancer detection
Preprocessing.py has the fns to convert the dicom files to pngs after resizzing and application of CLAHE enhacements.
Train.py is yolo model training. Config.yaml is being used here, it contains the class information of the detection task along with the repositories of the files in use.
Results.py develops test results dictionary and than prints results in the form of a confusion matrix 
The Dataset in use in VinDrMammo.
