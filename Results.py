#Making a Pridictions Disctionary

from ultralytics import YOLO
import glob
import os
import pandas as pd

# Read 'vindr.csv' file into a DataFrame
vindr_data = pd.read_csv('/content/drive/MyDrive/MISC/vindr.csv')

# Load a pretrained YOLOv8n model
model = YOLO('/content/drive/MyDrive/runs/detect/train4/weights/best.pt')

test_pngs = sorted(glob.glob('/content/drive/MyDrive/VinDr/test/images/*.png'))

# Initialize an empty list to store prediction information
predictions = []

# Initialize a dictionary to count cancer predictions for each class
cancer_counts = {}

# Loop through each image in test_pngs
for image_path in test_pngs:
    # Run inference on the current image
    # results = model(image_path, conf=0.2)
    results = model(image_path)
    # Count the cancer predictions for each class
    counts = {}
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in counts:
                counts[cls] = 1
            else:
                counts[cls] += 1

    # Determine the prediction (cancer or normal)
    if any(counts.values()):  # Check if there are any cancer detections
        prediction = "cancer"
    else:
        prediction = "normal"

    # Extract the image name
    image_name = os.path.basename(image_path)
    img_idvindr = image_name.split("_")[0]

    # Find all rows with the matching image ID in 'vindr.csv'
    rows_for_image_id = vindr_data[vindr_data['image_id'] == img_idvindr]

    # Check if any findings are available for the image ID
    if not rows_for_image_id.empty:
        # Get the unique cancer types for this image ID
        finding = set(rows_for_image_id['findings'])
        predictions.append({"image_name": image_name, "prediction": prediction, "finding": "cancer"})
    else:
        predictions.append({"image_name": image_name, "prediction": prediction, "finding": "normal"})


for prediction_info in predictions:
    print(f"Image: {prediction_info['image_name']}, Prediction: {prediction_info['prediction']}, Finding: {prediction_info['finding'] }")

# Count the total number of cancer and normal predictions
total_cancer_predictions = sum(1 for prediction_info in predictions if prediction_info['prediction'] == 'cancer')
total_normal_predictions = sum(1 for prediction_info in predictions if prediction_info['prediction'] == 'normal')


# Print the total counts of cancer and normal predictions
print(f"Total Cancer Predictions: {total_cancer_predictions}")
print(f"Total Normal Predictions: {total_normal_predictions}")

# Confusion Matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Extract the true labels (actual findings) and predicted labels from the predictions array
true_labels = [info['finding'] for info in predictions]
predicted_labels = [info['prediction'] for info in predictions]

# Define the class labels
classes = ['normal', 'cancer']

# Calculate the confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels, labels=classes)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, values_format='d')

plt.title('Confusion Matrix')
plt.show()
