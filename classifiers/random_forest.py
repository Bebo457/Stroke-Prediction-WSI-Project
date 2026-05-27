import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


RANDOM_STATE = 42

# WCZYTANIE DANYCH I PODZIAŁ
df = pd.read_csv("../dataset/dataset_prep.csv")

X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train = 60%, reszta = 40%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Rozmiary zbiorów:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nRozkład klas przed SMOTE:")
print("Train:")
print(y_train.value_counts())
print("\nValidation:")
print(y_val.value_counts())
print("\nTest:")
print(y_test.value_counts())

# PIPELINE: SCALER + PCA + SMOTE + RANDOM FOREST
# StandardScaler jest potrzebny przed PCA
# PCA uczy się tylko na zbiorze treningowym.
# SMOTE jest stosowany tylko podczas trenowania, nie na validation/test.

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(
        n_components=0.95,
        random_state=RANDOM_STATE
    )),
    ("smote", SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=3
    )),
    ("rf", RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# RANDOM FOREST + GRID SEARCH
param_grid = {
    "rf__n_estimators": [300, 400],
    "rf__max_depth": [None, 20, 30],
    "rf__min_samples_leaf": [2, 3, 4],
    "rf__min_samples_split": [5, 10],
    "rf__max_features": ["log2"],
    "rf__class_weight": [
        {0: 1, 1: 2},
        {0: 1, 1: 3}
    ]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
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
    "param_rf__n_estimators",
    "param_rf__max_depth",
    "param_rf__min_samples_leaf",
    "param_rf__min_samples_split",
    "param_rf__max_features",
    "param_rf__class_weight"
]

best_model = grid_search.best_estimator_

print("\nNajlepsze parametry:")
print(grid_search.best_params_)

print("\nNajlepszy ROC AUC CV:")
print(grid_search.best_score_)

# Informacja o PCA
pca_model = best_model.named_steps["pca"]

print("\n=== PCA ===")
print("Liczba komponentów PCA:", pca_model.n_components_)
print("Wyjaśniona wariancja:", pca_model.explained_variance_ratio_.sum())

# DOBÓR THRESHOLDU NA WALIDACJI
val_prob = best_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.05, 0.96, 0.05):
    val_pred_threshold = (val_prob >= threshold).astype(int)

    precision = precision_score(y_val, val_pred_threshold, pos_label=1, zero_division=0)
    recall = recall_score(y_val, val_pred_threshold, pos_label=1, zero_division=0)
    f1 = f1_score(y_val, val_pred_threshold, pos_label=1, zero_division=0)
    f2 = fbeta_score(y_val, val_pred_threshold, beta=2, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_val, val_pred_threshold)
    tn, fp, fn, tp = cm.ravel()

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

threshold_df = pd.DataFrame(threshold_results)

best_threshold_row = threshold_df.sort_values(by="f2", ascending=False).iloc[0]
best_threshold = best_threshold_row["threshold"]

print("\nNajlepszy threshold według F2-score:")
print(best_threshold_row)

# EWALUACJA MODELU
def evaluate_model(model, X, y, dataset_name, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
    f2 = fbeta_score(y, y_pred, beta=2, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y, y_prob)

    print(f"\n=== METRYKI DLA {dataset_name} ===")
    print(f"Threshold:          {threshold:.2f}")
    print(f"Precision stroke=1: {precision:.4f}")
    print(f"Recall stroke=1:    {recall:.4f}")
    print(f"F1-score stroke=1:  {f1:.4f}")
    print(f"F2-score stroke=1:  {f2:.4f}")
    print(f"ROC AUC:            {roc_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))

    return y_pred, y_prob, precision, recall, f1, f2, roc_auc

# WALIDACJA I TEST Z WYBRANYM THRESHOLDEM
val_pred, val_prob, val_precision, val_recall, val_f1, val_f2, val_roc_auc = evaluate_model(
    best_model,
    X_val,
    y_val,
    "VALIDATION",
    threshold=best_threshold
)

test_pred, test_prob, test_precision, test_recall, test_f1, test_f2, test_roc_auc = evaluate_model(
    best_model,
    X_test,
    y_test,
    "TEST",
    threshold=best_threshold
)

# CONFUSION MATRIX - TEST
cm = confusion_matrix(y_test, test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No stroke", "Stroke"]
)

disp.plot(
    cmap=plt.cm.Blues,
    values_format="d"
)

plt.title(f"Confusion Matrix - RF SMOTE Train PCA, threshold={best_threshold:.2f}")
plt.savefig("plots/RF_smote_train_PCA_confusion_matrix_threshold.png", dpi=300, bbox_inches="tight")
plt.close()

# ROC CURVE - TEST
fpr, tpr, _ = roc_curve(y_test, test_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {test_roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest SMOTE Train PCA")
plt.legend()
plt.grid()

plt.savefig("plots/RF_smote_train_PCA_roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# WAŻNOŚĆ KOMPONENTÓW PCA
rf_model = best_model.named_steps["rf"]

component_names = [f"PC{i+1}" for i in range(pca_model.n_components_)]

component_importances = pd.DataFrame({
    "component": component_names,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== WAŻNOŚĆ KOMPONENTÓW PCA ===")
print(component_importances.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(component_importances["component"], component_importances["importance"])
plt.xlabel("Importance")
plt.ylabel("PCA Component")
plt.title("Component Importance - Random Forest SMOTE Train PCA")
plt.gca().invert_yaxis()
plt.grid(axis="x")

plt.savefig("plots/RF_smote_train_PCA_component_importance.png", dpi=300, bbox_inches="tight")
plt.close()

# ZAPIS MODELU
model_bundle = {
    "model": best_model,
    "threshold": best_threshold,
    "best_params": grid_search.best_params_,
    "best_cv_score": grid_search.best_score_,
    "pca_n_components": pca_model.n_components_,
    "pca_explained_variance": pca_model.explained_variance_ratio_.sum(),
    "features": list(X_train.columns)
}

joblib.dump(model_bundle, "random_forest_smote_train_PCA_classifier.pkl")