import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score
)

# 1. Load Dataset
data = pd.read_csv("11_mushroom_edibility.csv")

# 2. Separate Features and Target
X = data.drop(columns=["SampleID", "Class"])
y = (data["Class"].str.lower() == "poisonous").astype(int)
# Edible: 0, Poisonous: 1

# 3. Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Pipeline setup with One-Hot Encoding and Decision Tree
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)
    ]
)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", decision_tree)
])

# 5. Train Model & Predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Confusion Matrix & Metrics
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = TN / (TN + FP)

print("----- Confusion Matrix -----")
print(cm)
print(f"TN: {TN} | FP: {FP} | FN: {FN} | TP: {TP}")

print("\n----- Evaluation Metrics -----")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Error Rate  : {error:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")

print("\n----- Classification Report -----")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Edible", "Poisonous"]
))
