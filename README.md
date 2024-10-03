# Machine Learning Model for Haul Truck Operation State Prediction

A hackathon submission for the OES ML challenge 2024 (https://www.hackerearth.com/challenges/new/competitive/oes-ml-challenge2024/).

## 🎒 Tech Stack


![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)


## Overview
This project focuses on classifying the operational state of haul trucks based on telemetry data from open-pit mining operations. We performed feature engineering to extract valuable patterns and trained machine learning models to predict the `operation_kind_id`. The solution utilizes both traditional machine learning techniques (like XGBoost) and advanced neural network architectures.

## Dataset
- **Telemetry Data**: Contains real-time data about the vehicles (e.g., speed, latitude, longitude, altitude, etc.).
- **Operations Data**: Contains labels of the operational state (e.g., `operation_kind_id`).
- **Validation Data**: Telemetry data used for validation and prediction submission.

## Key Columns
- **Telemetry Columns**: `speed_gps`, `lat`, `lon`, `alt`, `direction`, `accel_forward_nn`, `accel_braking_nn`, `accel_angular_nn`, etc.
- **Operations Column**: `operation_kind_id`

## Approach
1. **Data Merging**:
   - The telemetry and operations datasets were merged using time intervals (`create_dt`, `start_time`, and `end_time`), assigning the appropriate `operation_kind_id` to telemetry entries that fall within a specific operation period.
   
2. **Feature Engineering**:
   - **Time-based Features**: Extracted hour, day, and month from the `create_dt` column.
   - **Speed and Acceleration Features**: Calculated speed changes and acceleration differences.
   - **Lag and Lead Features**: Created lag/lead columns to capture the temporal dependencies in telemetry.
   - **Rolling Statistics**: Computed rolling mean and standard deviation for key features (e.g., speed, acceleration).
   - **Interaction Features**: Multiplied telemetry features to generate interaction terms (e.g., speed × acceleration).
   - **Distance Calculation**: Using the Haversine formula to calculate the distance traveled between consecutive GPS points (latitude and longitude).

Ultimately, most engineered features don't provide better accuracy with the exception  of Distance (Haversine formula) and GPS Speed.

3. **Modeling**:
   - **XGBoost Classifier**: XGBoost was selected for its efficiency in handling structured data with a focus on class imbalance.
   - **Neural Networks (Optional)**: We also experimented with LSTM and GRU architectures to capture sequential patterns in the telemetry data.

To find the best performing model, AutoML by H2O Wave is utilised to try Neural networks, XGBoost, Distributed Random forest and various ensemble combinations of these ML models (except neural networks). 

4. **Handling Class Imbalance**:
   - The dataset exhibited class imbalance, which was handled using a custom PyTorch resampler that undersamples majority classes and oversample minority classes. However, the patterns which were obtained in the imbalanced dataset are lost and hence, resampled data reduces accuracy and score.

5. **Performance Evaluation**:
   - The model was evaluated based on accuracy, and we aimed to improve it to over 60% by experimenting with more complex features and neural network structures.

6. **Challenges**:
   - **Class Distribution**: Some classes were underrepresented in the training data, making classification difficult.
   - **Memory Constraints**: Neural network models faced memory issues on large datasets, so XGBoost was ultimately used as a fallback.

## Tools Used:
- **Libraries**:
   - `pandas`: For data manipulation and processing.
   - `numpy`: For numerical operations and feature creation.
   - `matplotlib`, `seaborn`: For data visualization.
   - `XGBoost`: For classification.
   - `scikit-learn`: For data preprocessing and model evaluation.
   - `TensorFlow` / `Keras` (Optional): For neural network experimentation.

## Prod Model
The directory named Deep Learning has a MOJO model that gave us a score of 69.75%. This MOJO model can be used for predictions in production environment. 
For real time prediction this model can be used in an architecture with Apache Kafka handling the streaming of data and the model making predictions utilising Apache Flink. 

## Project Organisation

    .
    ├── DeepLearning_grid_1_AutoML_3_20241003_130344_model_1    <- Production model
    ├── README.md   <- guide for devs
    ├── auto.csv    <- formatted prediciton file
    ├── automl.ipynb    <- notebook outlining implementation of H2O AutoML
    ├── federated learning .ipynb   <- notebook to try and mimic the Deep Learning model obtained in H2O AutoML
    ├── merging.ipynb   <- Notebook to merge original datasets for traininng
    ├── operations_labels_training.csv <- input file in hackathon
    ├── pediction-formatter.ipynb   <- notebbok to format H2O AutoML model predicitons into requisite format
    ├── telemetry_for_operations_training.csv <- input file in hackathon
    ├── telemetry_for_operations_validation.csv <- input file in hackathon
    └── xggs.ipynb <- notebook for XGBoost training and prediction output.


## 👨‍💻 Authors

[![Static Badge](https://img.shields.io/badge/aravindan2-red?logo=GitHub&link=https%3A%2F%2Fgithub.com%2Faravindan2)
](https://www.github.com/aravindan2)
[![Static Badge](https://img.shields.io/badge/Rajkanwars15-yellow?logo=GitHub&link=https%3A%2F%2Fgithub.com%2FRajkanwars15)
](https://www.github.com/rajkanwars15)

# Submitted by Team ACRUX. Last checked Rank- 41. Last checked score- 69.75% on Deep Learning based classifier.