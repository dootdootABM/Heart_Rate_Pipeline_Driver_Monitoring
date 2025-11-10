# Drowsiness Detection using HRV and Machine Learning
This project aims to detect and predict drowsiness levels by analyzing Heart Rate Variability (HRV) data. We utilize machine learning models, including Random Forest and XGBoost, to classify drowsiness states based on features extracted from physiological signals.
## Getting Started
Follow these instructions to set up your local environment and run the project.
### 1. Prerequisites
This project requires Python 3.11.9. Please ensure you have this specific version installed on your system.
### 2. Install Dependencies
Install all required packages by running the following command in your terminal. (A requirements.txt file is recommended for larger projects).
```
pip install pandas matplotlib seaborn neurokit2 datasets transformers torch
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
1. `load_data.py`: Loads raw data and generates exploratory data analysis (EDA) graphs (histograms, box plots, etc.).
2. `feature_engineering.py`: Processes signals, extracts HRV features using neurokit2, and outputs a new CSV file for training.
3. `data_split.py`: Splits the feature dataset into unbiased Training, Testing, and Validation sets.
4. `train_model.py`: Implements and trains the primary Random Forest classifier and evaluates its performance.
5. `train_model_llm.py`: _(EXPERIMENTAL)_ Fine-tunes a Llama 3.1 model. Not a core part of the project.

### 5. Project Agenda & Roadmap
This section outlines the immediate next steps for the project.
1. Refine Random Forest. Improve the train_model.py implementation with better code structure and hyperparameter tuning.
2. Feature Enhancement & PCA. Combine HRV values with timestamps. Use Principal Component Analysis (PCA) to reduce dimensionality.
3. Implement XGBoost. Create a new script (e.g., train_model_xgboost.py) to train an XGBoost model for comparison.
4. Benchmark & Visualize. Develop a comparison chart (Accuracy, F1-Score, etc.) and generate confusion matrices and feature importance plots.


