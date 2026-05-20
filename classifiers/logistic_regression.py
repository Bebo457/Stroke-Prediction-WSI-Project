import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression  # PODMIANA: Regresja logistyczna
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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
# TRENOWANIE REGRESJI LOGISTYCZNEJ (Z BALANSEM)
# =========================

print("Trenowanie modelu regresji logistycznej z balansowaniem klas...")

# Zmiana modelu na logistyczny z wbudowanym balansem klas
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("\nWspółczynniki regresji:")
# model.coef_[0] zawiera współczynniki dla klasy pozytywnej
for feature, coef in zip(X_train.columns, model.coef_[0]):
    print(f"  {feature}: {coef:.6f}")
print(f"Wyraz wolny (intercept): {model.intercept_[0]:.6f}")

# =========================
# WALIDACJA
# =========================

# Wyciągamy prawdopodobieństwo klasy 1 (udar), co daje wartości od 0.0 do 1.0
val_predictions = model.predict_proba(X_val)[:, 1]
val_predictions_binary = (val_predictions >= 0.5).astype(int)

print("\n=== WYNIKI WALIDACJI ===")
print("Mean squared error:", mean_squared_error(y_val, val_predictions))
print("Mean absolute error:", mean_absolute_error(y_val, val_predictions))
print("R2 score:", r2_score(y_val, val_predictions))
print("Accuracy (po zaokrągleniu 0.5):", accuracy_score(y_val, val_predictions_binary))

# =========================
# TEST
# =========================

test_predictions = model.predict_proba(X_test)[:, 1]
test_predictions_binary = (test_predictions >= 0.5).astype(int)

print("\n=== WYNIKI TESTU ===")
print("Mean squared error:", mean_squared_error(y_test, test_predictions))
print("Mean absolute error:", mean_absolute_error(y_test, test_predictions))
print("R2 score:", r2_score(y_test, test_predictions))
print("Accuracy (po zaokrągleniu 0.5):", accuracy_score(y_test, test_predictions_binary))

print("\nClassification Report:")
print(classification_report(y_test, test_predictions_binary))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions_binary))

# =========================
# ZAPIS MODELU
# =========================

# Zapisujemy nowy model logistyczny
joblib.dump(model, "logistic_regression_stroke_model.pkl")
print("\nModel zapisany jako logistic_regression_stroke_model.pkl")