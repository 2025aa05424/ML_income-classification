# Income Classification using Machine Learning Models  
## Submission by 2025AA05424

## Problem Statement

The objective of this project is to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes. This is a binary classification problem using the Adult Census Income dataset.

---

## Dataset Description

The Adult Census Income dataset is sourced from the UCI Machine Learning Repository and is publicly available on Kaggle.

After preprocessing:
- Total instances: 30,162
- Total features (after one-hot encoding): 96
- Target variable: `income`
    - 0 → <=50K
    - 1 → >50K

Missing values were handled by removing rows with unknown entries.  
Categorical variables were encoded using one-hot encoding.  

The dataset is moderately imbalanced (~75% <=50K and ~25% >50K).

---

## Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------|----------|------|----------|--------|------|------|
| Logistic Regression | 0.8548 | 0.9132 | 0.7504 | 0.6245 | 0.6817 | 0.5928 |
| Decision Tree | 0.8155 | 0.7528 | 0.6299 | 0.6278 | 0.6289 | 0.5061 |
| KNN | 0.8258 | 0.8497 | 0.6667 | 0.6005 | 0.6319 | 0.5194 |
| Naive Bayes | 0.7910 | 0.8369 | 0.6681 | 0.3189 | 0.4317 | 0.3559 |
| Random Forest | 0.8550 | 0.9104 | 0.7467 | 0.6318 | 0.6845 | 0.5946 |
| XGBoost | 0.8722 | 0.9338 | 0.7903 | 0.6625 | 0.7208 | 0.6429 |

---

## Observations on Model Performance

### Logistic Regression
Performed strongly with high AUC and balanced precision-recall. It handles this dataset well due to relatively linear separability of features.

### Decision Tree
Showed moderate performance but lower AUC compared to ensemble methods. Single decision trees may overfit and produce less stable predictions.

### K-Nearest Neighbors
Achieved decent performance but slightly lower than Logistic Regression. Performance depends on proper feature scaling.

### Naive Bayes
Fast and computationally efficient but weakest performer. The low recall indicates difficulty in capturing higher income cases due to the independence assumption between features.

### Random Forest
Improved performance compared to a single Decision Tree. Ensemble averaging reduced overfitting and improved stability.

### XGBoost
Best performing model across most metrics. Gradient boosting effectively captures feature interactions and improves generalization.

---

## Streamlit Deployment

The trained models were deployed using a Streamlit web application.

The application allows users to:
- Upload a CSV file  
- Select a classification model  
- View evaluation metrics  
- View confusion matrix  

The app is deployed using Streamlit Community Cloud.
