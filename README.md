## Description
This repository contains a comprehensive analysis of EEG (Electroencephalography) data using various deep learning models to distinguish between healthy individuals and long COVID patients. The project employs convolutional neural networks (CNN), long short-term memory networks (LSTM), and their combinations to extract meaningful patterns from EEG signals and achieve high classification accuracy.

## Features
- **Data Loading**: Efficiently load EEG data from CSV files.
- **Preprocessing**: Standardize and shuffle the data to prepare it for model training.
- **Modeling**: Implementation of CNN, CNN-LSTM, ConvLSTM, and BiLSTM models for classification.
- **Evaluation**: Generate confusion matrices and classification reports to evaluate model performance.
- **Visualization**: Plot training history to visualize model accuracy and loss over epochs.

### Files and Directories
- **src/**: Contains all the source code for data loading, preprocessing, modeling, and utilities.
  - **data_loader.py**: Functions to load data from CSV files.
  - **preprocessing.py**: Functions to preprocess and standardize the data.
  - **utils.py**: Utility functions such as plotting the confusion matrix.
  - **cnn_lstm_model.py**: Definition of the CNN-LSTM model.
  - **convlstm_model.py**: Definition of the ConvLSTM model.
  - **bilstm_model.py**: Definition of the BiLSTM model.
  - **cnn_model.py**: Definition of the CNN model.
  - **main.py**: Main script to run the entire analysis, from data loading to model evaluation.
- **README.md**: Detailed description and instructions for the project.
- **requirements.txt**: List of Python dependencies required for the project.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Requirements
Ensure you have the following packages installed:
- numpy
- pandas
- matplotlib
- scikit-learn
- keras
- tensorflow
- keras-self-attention

Install the requirements with: pip install -r requirements.txt
