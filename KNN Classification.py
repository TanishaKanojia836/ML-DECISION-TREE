# KNN Classification - 11_mushroom_edibility.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    recall_score, precision_score, f1_score, roc_auc_score, roc_curve
)

# 1. Load Dataset
data = pd.read_csv("11_mushroom_edibility.csv")

# 2. Separate Features and Target (Drop SampleID identifier)
X = data.drop(columns=["SampleID", "Class"])
y = (data["Class"].str.lower() == "poisonous").astype(int)  # Edible: 0, Poisonous: 1

# 3. Train-Test Split (80:20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Pipeline Setup: One-Hot Encoding + KNN (k=5)
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)]
)

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)  # Euclidean distance

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", knn)
])

# 5. Fit & Predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Confusion Matrix & Metrics
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = TN / (TN + FP)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("----- Confusion Matrix -----")
print(cm)
print(f"TN: {TN} | FP: {FP} | FN: {FN} | TP: {TP}")

print("\n----- Evaluation Metrics -----")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Error Rate  : {error:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")

# 7. Classification Report
print("\n----- Classification Report -----")
print(classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"]))

# 8. 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("----- 5-Fold Cross-Validation -----")
print("Fold Scores :", np.round(cv_scores, 4))
print(f"Mean CV Acc : {cv_scores.mean():.4f}")
print(f"Std Dev     : {cv_scores.std():.4f}")

# 9. ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"KNN (AUC = {auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN Classifier")
plt.legend()
plt.tight_layout()
plt.show()