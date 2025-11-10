# Drowsiness Detection Pipeline: Analysis Report

**Group Project 2, Team 2: Heart-rate Monitoring**
*10.11.2025*

---

## 1. Project Overview

### Problem Statement
Develop the heart rate sensor pipeline using base heart rate and HRV and build a history context using external factors to resolve ambiguity in sensor data to safely presume fatigue/drowsiness.

### Hypothesis
* Using LLMs to predict the Karolinska-Sleepiness Scale based on time of day, duration of drive, Standard Deviation in Lane Position and Speed, and a brief history context can increase reasoning accuracy.
* An LLM's accuracy will be higher when provided with a structured history of prior predictions and sensor summaries compared to a stateless LLM.
* Summarized historical context will achieve similar or better accuracy than raw full-history prompts while reducing token usage and latency.
* Models with conflict/reliability context will recover baseline accuracy faster after simulated sensor dropout compared to stateless LLMs.

---

## 2. Original Data Analysis (EDA)

Before feature engineering, we analyzed the raw sensor data to understand its basic properties and relationships.

### Feature Distributions and Outliers

* **Distributions:** The `drowsiness` plot clearly shows our three target classes (0.0 for Alert, 1.0 for Drowsy, 2.0 for Very Drowsy). The raw sensor data (`ppgGreen`, `ppgRed`, `ppgIR`) is noisy and widely distributed. `heartRate` is roughly normal but with a long tail.
* **Boxplots:** The boxplots confirm the `heartRate` distribution and highlight several outliers, particularly on the high end.

![Distributions of Numerical Features](Original_Data_Analysis\numerical_distributions.png)
![Box Plots of Numerical Features](Original_Data_Analysis\numerical_boxplots.png)

### Correlation Analysis

The correlation heatmap reveals the most important relationships in the raw data.

**Key Findings:**
* **`heartRate` vs `drowsiness` (-0.68):** This is the strongest predictive signal in the raw data. It shows a strong negative correlation, meaning as **heart rate decreases**, the **drowsiness level increases**.
* **`ppgRed`/`ppgIR` vs `drowsiness` (-0.39 / -0.41):** A moderate negative correlation, likely tied to the same physiological response as the heart rate.

![Correlation Heatmap](Original_Data_Analysis\correlation_heatmap.png)

---

## 3. Feature Engineering & PCA

The raw signals were processed into sliding windows, and a set of 90+ Heart Rate Variability (HRV) features were engineered for each window. We used Principal Component Analysis (PCA) to understand the new feature space.

### PCA: Explained Variance
This plot shows that the feature set has high dimensionality. The first principal component only explains 35% of the variance. To capture 95% of the variance, we would need approximately 25-30 components.

![PCA Cumulative Explained Variance](HRV_PCA\pca_explained_variance.png)

### PCA: 2D Visualization
Plotting the first two principal components shows that the three drowsiness classes (0.0, 1.0, 2.0) are heavily overlapped. There is no clear linear boundary that can separate them. This confirms that we need more complex, non-linear models (like Random Forest or XGBoost) to find a good decision boundary.

![2D PCA Visualization](HRV_PCA\pca_2d_visualization.png)

---

## 4. Engineered Data Overview (Model Input)

This is a look at the final dataset that was fed into our models *after* windowing and feature engineering.

* **Label Distribution:** The final dataset is imbalanced. Class 0.0 (Alert) is the most common, which is typical for real-world driving data. This imbalance is handled in the models using `class_weight='balanced'` or similar techniques.
* **Feature Distributions:** Many of the engineered HRV features are heavily right-skewed (e.g., `HRV_RMSSD`). Tree-based models like Random Forest and XGBoost are excellent for this type of data, as they are not sensitive to feature scale or distribution.

![Engineered Label Distribution](Results_XGBoost\label_distribution.png)
![Key Feature Distributions](Results_XGBoost\key_feature_distributions.png)

---

## 5. Model Comparison: Random Forest vs. XGBoost

We trained two powerful tree-based models on the engineered HRV features.

### Random Forest

* **Performance:** The confusion matrix shows that Random Forest is very effective at identifying Class 0.0 (Alert) and Class 2.0 (Very Drowsy). However, it struggles with the intermediate Class 1.0, frequently misclassifying it as 0.0 (Alert).
* **Key Features:** The most important features for this model are `HRV_MCVNN`, `HRV_MadNN`, and `HRV_pNN20`—all metrics related to the variability and spread of heartbeats.

![Random Forest Confusion Matrix](Results_RandomForest\rf_confusion_matrix_heatmap.png)
![Random Forest Feature Importances](Results_RandomForest\rf_feature_importances.png)

### XGBoost (Extreme Gradient Boosting)

* **Performance:** XGBoost shows a very similar pattern. It also struggles with Class 1.0, misclassifying 11 instances as 0.0. In this specific test set, it achieved perfect classification for Class 2.0.
* **Key Features:** XGBoost overwhelmingly agrees with Random Forest. The `HRV_MadNN` (Median Absolute Deviation of NN intervals) is by far the most dominant feature for prediction. Other variability metrics like `HRV_SDNNi1` and `HRV_pNN20` are also ranked highly.

![XGBoost Confusion Matrix](Results_XGBoost\confusion_matrix_heatmap.png)
![XGBoost Feature Importances](Results_XGBoost\xgb_feature_importances.png)

---

## 6. Conclusion

1.  **Raw Data:** The raw `heartRate` signal is a strong predictor on its own, showing a **-0.68** correlation with drowsiness.
2.  **Engineered Features:** Engineering HRV features allows for more nuanced predictions. The dataset is high-dimensional and non-linear, justifying the use of models like Random Forest and XGBoost.
3.  **Model Performance:** Both models excel at distinguishing **Alert (0.0)** from **Very Drowsy (2.0)**. Their primary weakness is classifying the ambiguous **Drowsy (1.0)** state, which they often confuse with the Alert state.
4.  **Key Predictors:** Both models confirm that **Heart Rate Variability** metrics are the most important features. Specifically, `HRV_MadNN` (Median Absolute Deviation) is the single most powerful predictor in the dataset. This suggests that the *consistency* of the heartbeat (or lack thereof) is a stronger signal than the heart rate itself.
```eof