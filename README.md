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
| Random Forest | 0.8571 | 0.9161 | 0.8181 | 0.5479 | 0.6563 | 0.5888 |
| XGBoost | 0.8722 | 0.9338 | 0.7903 | 0.6625 | 0.7208 | 0.6429 |

---

## Observations on Model Performance

### Logistic Regression  
Logistic Regression achieved strong performance with 85.48% accuracy and an AUC of 0.9132, indicating strong class separability. Precision (0.7504) is higher than recall (0.6245), meaning the model is relatively conservative in predicting high-income (>50K) cases — when it predicts positive, it is usually correct, but it misses some true high-income individuals. The MCC (0.5928) confirms stable performance even under class imbalance. Logistic Regression performs well because several demographic and employment features exhibit approximately linear relationships with the log-odds of income class.

### Decision Tree  
The Decision Tree achieved 81.55% accuracy with the lowest AUC (0.7528) among all models, indicating weaker probability ranking ability. Precision and recall are balanced (~0.63), suggesting unbiased but moderate-quality predictions. The MCC of 0.5061 reflects weaker correlation with true labels compared to ensemble methods. Single decision trees are prone to overfitting and sensitive to data variations, which explains the performance gap compared to ensemble-based approaches.

### K-Nearest Neighbors (KNN)  
KNN achieved 82.58% accuracy with an AUC of 0.8497, showing moderate discriminative ability. Precision (0.6667) slightly exceeds recall (0.6005), meaning some high-income cases are missed. The model's performance depends heavily on the choice of k and proper feature scaling. Since KNN relies on distance calculations across all training samples, prediction can be computationally expensive. The moderate results indicate that neighborhood-based classification captures local patterns but does not outperform more structured models.

### Naive Bayes  
Naive Bayes was the weakest performer with 79.10% accuracy and a notably low recall (0.3189), meaning it correctly identifies only about 32% of actual high-income individuals. This results in the lowest F1 score (0.4317) and MCC (0.3559). The poor recall stems from the strong feature independence assumption, which does not hold well for demographic data where features like education, occupation, age, and working hours are correlated. Although computationally efficient and fast to train, Naive Bayes sacrifices predictive quality in this dataset.

### Random Forest (Ensemble)  
Random Forest achieved strong performance with 85.71% accuracy and an AUC of 0.9161, closely matching Logistic Regression. It achieved the highest precision (0.8181) among all models, meaning its positive predictions are highly reliable. However, recall (0.5479) is lower, indicating it misses a substantial portion of high-income cases. The ensemble of multiple decision trees reduces overfitting observed in a single Decision Tree and provides stable, robust predictions. Random Forest balances accuracy and robustness effectively.

### XGBoost (Ensemble)  
XGBoost was the best-performing model across all evaluation metrics, achieving 87.22% accuracy, the highest AUC (0.9338), the best F1 score (0.7208), and the highest MCC (0.6429). It demonstrates an excellent balance between precision (0.7903) and recall (0.6625), making it the most effective model at identifying high-income individuals without sacrificing overall accuracy. The gradient boosting framework iteratively corrects previous errors and captures complex non-linear feature interactions. Due to its superior performance across all metrics, XGBoost is the most suitable model for deployment.

---

## Streamlit Deployment

The trained models were deployed using a Streamlit web application.

The application allows users to:
- Upload a CSV file  
- Select a classification model  
- View evaluation metrics  
- View confusion matrix  

The app is deployed using Streamlit Community Cloud.



