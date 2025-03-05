# Ad Click Prediction - Machine Learning Model

## Overview
This project focuses on building and optimizing a machine learning model to predict whether a user will click on an advertisement based on various features. The primary goal was to achieve an accuracy of at least 97%.

## Dataset
The dataset used is `advertising.csv`, which contains 1000 observations and 10 features, including:
- `Daily Time Spent on Site`
- `Age`
- `Area Income`
- `Daily Internet Usage`
- `Ad Topic Line`
- `City`
- `Male`
- `Country`
- `Timestamp`
- `Clicked on Ad` (target variable)

## Exploratory Data Analysis (EDA)
- Pair plots showed a reasonable spread of users who clicked and did not click on ads.
- Descriptive statistics revealed a large range of values across numerical features.
- PCA (Principal Component Analysis) was used to visualize data distribution.

## Data Preprocessing
- Standardization was applied to numerical features to handle large variations in data.
- One-hot encoding was used for categorical features.
- Time-based features (Hour, Minute) were extracted from the `Timestamp` column.

## Model Selection and Optimization
Several models were evaluated:
1. **Support Vector Machine (SVM)** with GridSearchCV for hyperparameter tuning
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

### Best Performing Model
- **SVM with RBF Kernel** using optimized parameters (`C=10, gamma=0.01`) achieved **96% accuracy**.
- Feature engineering (dropping `Male` and leveraging `Timestamp` details) further refined the model.
- Encoding `Time` with `TfidfTransformer` was tested but did not significantly improve performance.

## Final Results
- **Best accuracy achieved: 96-97%**
- **SVM outperformed Decision Trees and Random Forest in consistency.**
- **Further improvements could explore deep learning approaches or additional feature engineering.**

## Future Enhancements
- Experiment with neural networks.
- Feature selection using SHAP or mutual information.
- Collect more data to generalize the model performance.

---
This project highlights the iterative process of improving machine learning models through preprocessing, feature engineering, and hyperparameter tuning to maximize accuracy.

