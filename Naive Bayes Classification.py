# ============================================================
# NAIVE BAYES CLASSIFICATION - MUSHROOM DATASET
# ============================================================

# 1. Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import StratifiedKFold, cross_val_score


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv("11_mushroom_edibility.csv")

print("\n----- FIRST FIVE RECORDS -----")
print(data.head())

print("\n----- DATASET SHAPE -----")
print(data.shape)

print("\n----- COLUMN NAMES -----")
print(data.columns.tolist())

print("\n----- MISSING VALUES -----")
print(data.isnull().sum().sum())


# ============================================================
# 3. IDENTIFY TARGET COLUMN
# ============================================================

# Automatically find the target column

possible_targets = [
    "class",
    "Class",
    "target",
    "Target",
    "label",
    "Label",
    "edibility",
    "Edibility"
]

target_column = None

for col in possible_targets:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError(
        "Target column was not found. "
        "Check the column names printed above."
    )

print("\nTarget Column:", target_column)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(target_column, axis=1)

y_original = data[target_column]

print("\nTarget values:")
print(y_original.value_counts())


# ============================================================
# 5. CONVERT TARGET INTO NUMERICAL VALUES
# ============================================================

# If target contains e/p:
# e = Edible = 0
# p = Poisonous = 1

if set(y_original.astype(str).str.lower().unique()) == {"e", "p"}:

    y = (
        y_original.astype(str)
        .str.lower()
        .map({"e": 0, "p": 1})
    )

else:
    # If target contains edible/poisonous words

    y_text = y_original.astype(str).str.lower()

    if set(y_text.unique()).issubset(
        {"edible", "poisonous"}
    ):

        y = y_text.map({
            "edible": 0,
            "poisonous": 1
        })

    else:
        # If target is already numerical
        y = pd.factorize(y_original)[0]

print("\nEncoded target values:")
print(pd.Series(y).value_counts())


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n----- TRAIN TEST SPLIT -----")

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 7. ONE-HOT ENCODING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            X.columns
        )
    ]
)


# ============================================================
# 8. CREATE NAIVE BAYES MODEL
# ============================================================

naive_bayes = MultinomialNB()


# ============================================================
# 9. CREATE PIPELINE
# ============================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", naive_bayes)
])


# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nNaive Bayes model trained successfully.")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Probability of class 1 (Poisonous)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n----- CONFUSION MATRIX -----")
print(cm)


# Extract TN, FP, FN, TP

TN, FP, FN, TP = cm.ravel()

print("\nTN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)


# ============================================================
# 13. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

specificity = TN / (TN + FP)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n----- EVALUATION METRICS -----")

print("Accuracy     :", round(accuracy, 4))
print("Error Rate   :", round(error_rate, 4))
print("Precision    :", round(precision, 4))
print("Recall       :", round(recall, 4))
print("Specificity  :", round(specificity, 4))
print("F1 Score     :", round(f1, 4))
print("ROC-AUC      :", round(auc, 4))


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n----- CLASSIFICATION REPORT -----")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Edible", "Poisonous"],
        zero_division=0
    )
)


# ============================================================
# 15. 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n----- 5-FOLD CROSS VALIDATION -----")

print("CV Scores:")

for i, score in enumerate(cv_scores, start=1):
    print(
        "Fold", i,
        ":", round(score, 4)
    )

print(
    "\nMean CV Accuracy:",
    round(cv_scores.mean(), 4)
)

print(
    "Standard Deviation:",
    round(cv_scores.std(), 4)
)


# ============================================================
# 16. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label="Naive Bayes (AUC = {:.2f})".format(auc)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Naive Bayes"
)

plt.legend()

plt.show()


# ============================================================
# 17. FINAL RESULT
# ============================================================

print("\n======================================")
print("       NAIVE BAYES FINAL RESULT")
print("======================================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Mean CV Accuracy:",
    round(cv_scores.mean() * 100, 2),
    "%"
)

print(
    "ROC-AUC:",
    round(auc, 4)
)

print("======================================")