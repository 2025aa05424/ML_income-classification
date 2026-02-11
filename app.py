import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Income Classification App")

st.write("Upload a test CSV file and select a model to evaluate.")

# Load models
logistic = joblib.load("models/logistic.pkl")
decision_tree = joblib.load("models/decision_tree.pkl")
knn = joblib.load("models/knn.pkl")
naive_bayes = joblib.load("models/naive_bayes.pkl")
random_forest = joblib.load("models/random_forest.pkl")
xgboost = joblib.load("models/xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

models = {
    "Logistic Regression": logistic,
    "Decision Tree": decision_tree,
    "KNN": knn,
    "Naive Bayes": naive_bayes,
    "Random Forest": random_forest,
    "XGBoost": xgboost
}

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_choice = st.selectbox("Select Model", list(models.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "income" not in df.columns:
        st.error("CSV must contain 'income' column as target.")
    else:
        X = df.drop("income", axis=1)
        y = df["income"]

        # Encode target
        y = y.map({"<=50K": 0, ">50K": 1})

        # One-hot encode
        X = pd.get_dummies(X, drop_first=True)

        # Align columns with training
        model = models[model_choice]

        if model_choice in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

        st.subheader("Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("Precision:", precision_score(y, y_pred))
        st.write("Recall:", recall_score(y, y_pred))
        st.write("F1 Score:", f1_score(y, y_pred))
        st.write("AUC:", roc_auc_score(y, y_prob))
        st.write("MCC:", matthews_corrcoef(y, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, y_pred))
