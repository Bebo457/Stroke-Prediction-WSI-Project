# Dataset preprocessing for Random Forest Classifier

import os
import pandas as pd
import numpy as np

DATASET_PATH      = "dataset/data_no_outliers.csv"
PREPROCESSED_PATH = "dataset/dataset_prep_RF.csv"

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.replace("N/A", np.nan, inplace=True)

    df.drop(columns=["id"], inplace=True)
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    # Imputacja BMI medianą
    df["bmi"] = df["bmi"].astype(float)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Label encoding zmiennych binarnych
    df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})
    df["gender"]       = df["gender"].map({"Male": 1, "Female": 0})

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["Residence_type", "work_type", "smoking_status"],
                        drop_first=False, dtype=int)

    stroke = df.pop("stroke")

    return df, stroke


if not os.path.exists(PREPROCESSED_PATH):
    df, stroke = preprocess(DATASET_PATH)
    df_to_save = df.copy()
    df_to_save["stroke"] = stroke
    df_to_save.to_csv(PREPROCESSED_PATH, index=False)
    print(f"Zapisano: {PREPROCESSED_PATH}")
else:
    df_full = pd.read_csv(PREPROCESSED_PATH)
    stroke  = df_full.pop("stroke")
    df      = df_full
    print(f"Wczytano istniejący plik: {PREPROCESSED_PATH}")

print(f"\nRozmiar zbioru: {df.shape}")
print(f"Przypadki udaru: {stroke.sum()} ({stroke.mean()*100:.2f}%)")