import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             roc_auc_score)
import matplotlib.pyplot as plt

TRAIN_PATH = "dataset/train_smote_all.csv"
VALID_PATH = "dataset/val_smote_all.csv"

PLOTS_DIR = "saved_plots/XGBoost"

df_train = pd.read_csv(TRAIN_PATH)
df_valid = pd.read_csv(VALID_PATH)

X_train = df_train.drop(columns=["stroke"])
y_train = df_train["stroke"]

X_val  = df_valid.drop(columns=["stroke"])
y_val  = df_valid["stroke"]

print(f"Train: {X_train.shape} | udary: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Val:   {X_val.shape}   | udary: {y_val.sum()} ({y_val.mean()*100:.2f}%)")

model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
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
plt.plot(train_loss, label="Train",      color="#4A90D9")
plt.plot(val_loss,   label="Walidacja",  color="#E84040")
plt.xlabel("Iteracja (drzewo)")
plt.ylabel("Logloss")
plt.title("Błąd na zbiorze treningowym i walidacyjnym")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.savefig(f"{PLOTS_DIR}/learning_curve.png")

plt.show()


model.save_model("xgboost_model.json")
print("Model zapisany: xgboost_model.json")