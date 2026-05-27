import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score
)
import joblib
from sklearn.model_selection import learning_curve
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# =========================
# WCZYTANIE DANYCH
# =========================
train_df = pd.read_csv("../dataset/train_smote_all.csv")
val_df = pd.read_csv("../dataset/val_smote_all.csv")
test_df = pd.read_csv("../dataset/test_smote_all.csv")

X_train = train_df.drop("stroke", axis=1)
y_train = train_df["stroke"]

X_val = val_df.drop("stroke", axis=1)
y_val = val_df["stroke"]

X_test = test_df.drop("stroke", axis=1)
y_test = test_df["stroke"]

# =========================
# SVM + GRID SEARCH
# =========================

pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("svm", SVC(probability=True, class_weight="balanced"))
])

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", "auto"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Najlepsze parametry:", grid_search.best_params_)

# =========================
# PROBABILITY (ROC)
# =========================
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# =========================
# METRYKI KLASY 1 (STROKE)
# =========================
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)

print("\n=== METRYKI DLA STROKE (1) ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Stroke Prediction")

plt.savefig("plots/wykres_confusion_matrix_SVM.png", dpi=300, bbox_inches="tight")

# =========================
# ROC CURVE
# =========================
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stroke Prediction (SVM)")
plt.legend()

plt.savefig("plots/wykres_ROC_SVM.png", dpi=300, bbox_inches="tight")