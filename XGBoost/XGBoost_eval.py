import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             RocCurveDisplay)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

TEST_PATH = "dataset/test_smote_all.csv"

PLOTS_DIR = "saved_plots/XGBoost"

df_test = pd.read_csv(TEST_PATH)
X_test  = df_test.drop(columns=["stroke"])
y_test  = df_test["stroke"]

model = XGBClassifier()
model.load_model("xgboost_model.json")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Metryki
print(classification_report(y_test, y_pred, target_names=["0", "1"], digits=4))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

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

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot(cmap="Blues")
plt.title("Macierz pomyłek – XGBoost")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150)

plt.show()
