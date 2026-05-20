import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

# =========================
# WCZYTANIE DANYCH
# =========================

train_df = pd.read_csv("../dataset/train.csv")
val_df = pd.read_csv("../dataset/validation.csv")
test_df = pd.read_csv("../dataset/test.csv")

# =========================
# PODZIAŁ NA X i y
# =========================

X_train = train_df.drop("stroke", axis=1)
y_train = train_df["stroke"]

X_val = val_df.drop("stroke", axis=1)
y_val = val_df["stroke"]

X_test = test_df.drop("stroke", axis=1)
y_test = test_df["stroke"]

# =========================
# TRENOWANIE SVM
# =========================

print("Trenowanie modelu SVM...")

svm_model = SVC()

# Strojenie hiperparametrów
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    svm_model,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nNajlepsze parametry:")
print(grid_search.best_params_)

# =========================
# WALIDACJA
# =========================

val_predictions = best_model.predict(X_val)

print("\n=== WYNIKI WALIDACJI ===")
print("Accuracy:", accuracy_score(y_val, val_predictions))

# =========================
# TEST
# =========================

test_predictions = best_model.predict(X_test)

print("\n=== WYNIKI TESTU ===")
print("Accuracy:", accuracy_score(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# =========================
# ZAPIS MODELU
# =========================

joblib.dump(best_model, "svm_stroke_classifier.pkl")

print("\nModel zapisany jako svm_stroke_classifier.pkl")