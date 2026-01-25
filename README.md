# Drowsiness Detection using HRV and Machine Learning
This project aims to detect and predict drowsiness levels by analyzing Heart Rate Variability (HRV) data. We utilize machine learning models, including Random Forest and XGBoost, to classify drowsiness states based on features extracted from physiological signals.
## Getting Started
Follow these instructions to set up your local environment and run the project.
### 1. Prerequisites
This project requires Python 3.10+. Please ensure you have this specific version installed on your system.
### 2. Install Dependencies
Install all required packages by running the following command in your terminal. (A requirements.txt file is recommended for larger projects).
```
pip install numpy pandas scipy matplotlib seaborn neurokit2 scikit-learn xgboost joblib datasets transformers torch
```
Note: If you encounter a ModuleNotFoundError for another package, please pip install it. The os package is a built-in Python module and does not require a separate installation.

### 3. IMPORTANT: Update File Paths
Before running any script, you must update all hardcoded file paths to match your local system's directory structure. This includes paths for:
1. Input CSV datasets
2. Output CSV files (like the HRV features)
3. Any saved models or graphs

### 4. Project Structure
Here is an overview of the key scripts in this project.
File
Description
1. `main.py`: Loads raw data and generates exploratory data analysis (EDA) graphs (histograms, box plots, etc.); Processes signals, extracts HRV features, performs principal component analysis and outputs a new CSV file for training.
2. `train_rf_combined` : Trains on the dataset using Random Forest with most important features after reducing dimensionality. Outputs all result infomatics.
3. `train_xg_combined.py`: Sames as previous script but with XGBoost
4. `heartratenode.py`: Reference python script using in https://github.com/dootdootABM/drowsiness_detection_ros2.git as a ROS2 node for the heart rate pipeline for data collection

### 5. Project Agenda & Roadmap
This section outlines the immediate next steps for the project.
1. Uses Sifi Labs BioPoint sensor to measure PPG Green data through already present driver simulator setup at Technische Hochschule Ingolstadt.
2. The training dataset includes the most relevant HRV features and combines time context as drive duration (seconds) and time period of driving(Morning, Afternoon, Evening, Late Night)
3. Compares Random Forest and XGBoost as machine learning algorithms.
4. In future, more focus will be on more data collection and procuring good and diverse data
5. Later will implement training on principal component (1-4) datasets.


