Breast Cancer Detection with YOLOv8

This README provides an overview of how to use each script. Ensure that the necessary dependencies, such as YOLOv8, OpenCV, and other relevant libraries, are installed before running the scripts. Feel free to modify paths and configurations as needed.

Running on Local Machine
Requirements:

    Python 3.x
    12GB or higher NVIDIA GPU with CUDA support
    CUDA Toolkit
    Git (for cloning the repository)

Installation:

    Clone Repository:
    Open a terminal and clone this repository to your local machine:
    
    git clone https://github.com/Shehabov/breast-cancer-detection.git

    Install Dependencies:
    Navigate to the project directory and install the required Python packages:
    
    cd breast-cancer-detection
    pip install -r requirements.txt

    YOLOv8 Installation:
    Follow the installation instructions for YOLOv8 on the official YOLO repository.

    Data Preparation:
    Organize your dataset according to the specified structure and adjust paths in the preprocessing.py script accordingly.

Usage:

    Preprocessing:-
    Run the preprocessing script:
    python Preprocessing.py

    Training:-
    Train the YOLOv8 model:
    python Train.py

    Results Evaluation:-
    Evaluate the trained model:
    python Results.py

Running on Google Colab
Requirements:

    Google Colab Pro/Pro+ account
    Select Higher RAM (36GB or higher)
    Select TPU or A100 GPU

Usage:

    Open Google Colab Notebook:
    Open a new Colab notebook using your Google account.

    Clone Repository:
    Add a new code cell in the notebook and run the following commands:
    
    !git clone https://github.com/your-username/breast-cancer-detection.git

    Install Dependencies and Libraries
    
    Add another code cell and run:
    %cd breast-cancer-detection
    !pip install -r requirements.txt

    Data Preparation:
    Upload your dataset to Google Drive and adjust paths in the preprocessing.py script accordingly.

    Login Google Drive.
    
    Mount Google Drive in Colab:    
    from google.colab import drive
    drive.mount('/content/drive')

    Usage:
    Use the preprocessing, training, and evaluation scripts in separate code cells:

    !python Preprocessing.py
    !python Train.py
    !python Results.py


This repository contains 3 files for preprocessing, training and results evaluation in the context of breast cancer detection using YOLOv8.

    "Preprocessing.py" file focuses on preparing the dataset for training. Follow these steps to use the preprocessing script:
    
    Dataset Structure:
    Ensure that your dataset is organized with proper directory structure, and each image is associated with relevant yolo annotations.

    Configurations:
    Modify the configuration parameters within the config.yaml, such as file paths according to your dataset.

    Run the Script:
    Execute the script using a Python environment. This will generate the preprocessed data necessary for training.


    "Train.py" file is responsible for training the YOLOv8 model on the preprocessed dataset. Follow these steps to initiate the training process:
    
    Preprocessed Data:
    Ensure that the preprocessing step has been completed, and the preprocessed data is available in the specified directory.

    Configuration:
    Adjust hyperparameters, dataset paths and training settings in the script based on your requirements.

    Start Training:
    Execute the script to begin training the YOLOv8 model.


    "Results.py" file assesses the model's performance by generating and visualizing a confusion matrix. Follow these steps to evaluate the results:
    
    Trained Model:
    Make sure that the YOLOv8 model has been trained using the training.py script and the weights are available.

    Configuration:
    Adjust the script's configurations, such as file paths and class labels, to match your setup.

    Evaluate Results:
    Execute the script to generate and display the confusion matrix based on model predictions.


Feel free to customize these scripts and configurations to fit your specific dataset and work requirements.
