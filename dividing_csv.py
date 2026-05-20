import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# =========================
# WCZYTANIE DANYCH
# =========================
df = pd.read_csv("dataset/dataset_prep_PCA.csv")

X = df.drop("stroke", axis=1)
y = df["stroke"]

# =========================
# SPLIT: train / temp
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y   # ważne!
)

# =========================
# SPLIT: val / test
# =========================
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# =========================
# OVERSAMPLING TYLKO TRAIN
# =========================
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Rozkład po SMOTE:")
print(y_train_res.value_counts())

# =========================
# SKŁADANIE DATAFRAME
# =========================
train_df = pd.concat([X_train_res, y_train_res], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# =========================
# ZAPIS
# =========================
train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/validation.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

print("Podział + SMOTE zakończony")