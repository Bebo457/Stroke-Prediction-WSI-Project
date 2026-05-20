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
# 🔴 SMOTE NA CAŁYM DATASECIE (jak w artykule)
# =========================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Po SMOTE (CAŁY DATASET):")
print(pd.Series(y_res).value_counts())

# =========================
# SPLIT PO SMOTE
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_res, y_res,
    test_size=0.3,
    random_state=42,
    stratify=y_res
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# =========================
# ZAPIS
# =========================
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("dataset/train_smote_all.csv", index=False)
val_df.to_csv("dataset/val_smote_all.csv", index=False)
test_df.to_csv("dataset/test_smote_all.csv", index=False)

print("Zrobione (wersja artykułowa - leakage)")