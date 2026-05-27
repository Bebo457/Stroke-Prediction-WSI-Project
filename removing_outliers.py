import pandas as pd

# ── Ścieżki ───────────────────────────────────────────────────────────────────
INPUT_PATH  = "dataset/dataset.csv"
OUTPUT_PATH = "dataset/data_no_outliers.csv"
# ─────────────────────────────────────────────────────────────────────────────

BMI_UPPER = 50.0
BMI_LOWER = 10.0 

df = pd.read_csv(INPUT_PATH)

# =========================
# 1. UZUPEŁNIANIE BRAKÓW MEDIANĄ
# =========================
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

print("Braki danych uzupełnione medianą.")

# =========================
# 2. KONWERSJA BMI
# =========================
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

# jeśli nadal coś NaN po konwersji → też medianą
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# =========================
# 3. OUTLIERY BMI
# =========================
mask = (df["bmi"] < BMI_LOWER) | (df["bmi"] > BMI_UPPER)

for idx in df[mask].index:
    print(f"usuwam wiersz id={df.loc[idx, 'id']}  |  bmi={df.loc[idx, 'bmi']}")

df_clean = df[~mask]

# =========================
# 4. PODSUMOWANIE
# =========================
print(f"\nWierszy przed: {len(df)}  |  usuniętych: {mask.sum()}  |  po: {len(df_clean)}")

df_clean.to_csv(OUTPUT_PATH, index=False)