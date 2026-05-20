import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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
svm_model = SVC(probability=True, class_weight="balanced")

param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    svm_model,
    param_grid,
    cv=5,
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
disp.plot()
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

# =========================
# LEARNING CURVE
# =========================

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",   # potem zamienimy na loss
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# accuracy → loss
train_loss = 1 - train_scores.mean(axis=1)
val_loss = 1 - val_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_loss, label="Train Loss (1-accuracy)")
plt.plot(train_sizes, val_loss, label="Validation Loss (1-accuracy)")

plt.xlabel("Liczba próbek treningowych")
plt.ylabel("Loss")
plt.title("Learning Curve (Loss) - SVM Stroke Prediction")

plt.legend()
plt.grid()

plt.savefig("plots/wykres_learning_curve_loss_SVM.png", dpi=300, bbox_inches="tight")

# =========================
# ZAPIS MODELU
# =========================
joblib.dump(best_model, "svm_stroke_classifier.pkl")

print("\nModel zapisany.")