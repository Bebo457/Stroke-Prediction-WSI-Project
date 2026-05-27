import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             RocCurveDisplay,
                             precision_score,
                             recall_score,
                             accuracy_score)
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Możliwe wartości MODE: "SMOTE_all" | "SMOTE_train" | "SMOTE_train_PCA"
MODE = "SMOTE_train_PCA"

HERE         = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(HERE)
DATASET_PATH = os.path.join(ROOT, "dataset", "dataset_prep.csv")
MODEL_PATH   = os.path.join(HERE, "trained_models", "xgboost_model.json")

MODE_LABELS = {
    "SMOTE_all":       "SMOTE (cały zbiór)",
    "SMOTE_train":     "SMOTE (zbiór treningowy)",
    "SMOTE_train_PCA": "SMOTE + PCA (zbiór treningowy)",
}

if MODE not in MODE_LABELS:
    raise ValueError(f"Nieznany MODE: {MODE}")

LABEL = MODE_LABELS[MODE]

if MODE == "SMOTE_all":
    RESULTS_DIR = os.path.join(HERE, "results", "SMOTE_all")
elif MODE == "SMOTE_train":
    RESULTS_DIR = os.path.join(HERE, "results", "SMOTE_train")
elif MODE == "SMOTE_train_PCA":
    RESULTS_DIR = os.path.join(HERE, "results", "SMOTE_train_PCA")

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
X  = df.drop(columns=["stroke"])
y  = df["stroke"]

smote = SMOTE(random_state=42)

if MODE == "SMOTE_all":
    X, y = smote.fit_resample(X, y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    X_train, y_train = smote.fit_resample(X_train, y_train)

if MODE == "SMOTE_train_PCA":
    pca = PCA(n_components=0.95, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    print(f"PCA: zachowano {pca.n_components_} składowych (95% wariancji)")

print(f"Train (po SMOTE): {X_train.shape} | udary: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Val:              {X_val.shape}   | udary: {y_val.sum()} ({y_val.mean()*100:.2f}%)")

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.01,
    min_child_weight=1,
    gamma=0.1,
    reg_lambda=10,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

results = model.evals_result()
train_loss = results["validation_0"]["logloss"]
val_loss   = results["validation_1"]["logloss"]

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Strata na zbiorze treningowym", color="#4A90D9")
plt.plot(val_loss,   label="Strata na zbiorze walidacyjnym", color="#E84040")
plt.xlabel("Iteracja")
plt.ylabel("Wartość funkcji straty")
plt.title(f"Krzywa uczenia – {LABEL}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"), dpi=150)
plt.show()

y_pred      = model.predict(X_val)
y_pred_prob = model.predict_proba(X_val)[:, 1]

precision = precision_score(y_val, y_pred)
recall    = recall_score(y_val, y_pred)
accuracy  = accuracy_score(y_val, y_pred)
auc_roc   = roc_auc_score(y_val, y_pred_prob)

print(classification_report(y_val, y_pred, target_names=["0", "1"], digits=4))
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")

metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(classification_report(y_val, y_pred, target_names=["0", "1"], digits=4))
    f.write(f"\nAccuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"AUC-ROC:   {auc_roc:.4f}\n")
print(f"Metryki zapisane: {metrics_path}")

print("\nMacierz pomyłek:")
print(confusion_matrix(y_val, y_pred))

RocCurveDisplay.from_predictions(y_val, y_pred_prob, name=f"XGBoost – {LABEL}")
plt.title(f"Krzywa ROC – {LABEL}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ROC.png"), dpi=150)
plt.show()

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot(cmap="Blues")
plt.title(f"Macierz pomyłek – {LABEL}")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

model.save_model(MODEL_PATH)
print(f"Model zapisany: {MODEL_PATH}")
