import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score
)
import joblib
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

RANDOM_STATE = 42

# WCZYTANIE DANYCH
train_df = pd.read_csv("../dataset/train_smote_all.csv")
val_df = pd.read_csv("../dataset/val_smote_all.csv")
test_df = pd.read_csv("../dataset/test_smote_all.csv")

X_train = train_df.drop("stroke", axis=1)
y_train = train_df["stroke"]

X_val = val_df.drop("stroke", axis=1)
y_val = val_df["stroke"]

X_test = test_df.drop("stroke", axis=1)
y_test = test_df["stroke"]

print("Rozkład klas:")
print("Train:")
print(y_train.value_counts())
print("\nValidation:")
print(y_val.value_counts())
print("\nTest:")
print(y_test.value_counts())

# RANDOM FOREST + GRID SEARCH
rf_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [15, 20, None],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt"],
    "class_weight": [None]
}

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)

cols = [
    "mean_train_score",
    "mean_test_score",
    "param_n_estimators",
    "param_max_depth",
    "param_min_samples_leaf",
    "param_min_samples_split",
    "param_max_features",
    "param_class_weight"
]

best_model = grid_search.best_estimator_

print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepszy ROC AUC CV:", grid_search.best_score_)

# PROBABILITY (ROC)
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# METRYKI KLASY 1 (STROKE)
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n=== METRYKI DLA STROKE (1) ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(
    cmap=plt.cm.Blues,
    values_format="d"
)
plt.title("Confusion Matrix - Stroke Prediction (Random Forest)")

plt.savefig("plots/RF_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stroke Prediction (Random Forest)")
plt.legend()
plt.grid()

plt.savefig("plots/RF_roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

disp.plot(
    cmap=plt.cm.Blues,
    values_format="d"
)


# WAŻNOŚĆ CECH
feature_importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== WAŻNOŚĆ CECH ===")
print(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances["feature"], feature_importances["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.grid(axis="x")

plt.savefig("plots/RF_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

# ZAPIS MODELU
joblib.dump(best_model, "random_forest_classifier.pkl")
