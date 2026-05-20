import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             RocCurveDisplay)
import matplotlib.pyplot as plt

TEST_PATH = "dataset/test_smote_all.csv"

PLOTS_DIR = "saved_plots/XGBoost"

df_test = pd.read_csv(TEST_PATH)
X_test  = df_test.drop(columns=["stroke"])
y_test  = df_test["stroke"]

model = XGBClassifier()
model.load_model("xgboost_model.json")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Predykcja
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Metryki
print(classification_report(y_test, y_pred, target_names=["Brak udaru", "Udar"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")

# Macierz pomyłek
print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred))

# Krzywa ROC
RocCurveDisplay.from_predictions(y_test, y_pred_prob, name="XGBoost")
plt.title("Krzywa ROC – XGBoost")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.savefig(f"{PLOTS_DIR}/ROC.png")

plt.show()

