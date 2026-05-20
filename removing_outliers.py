import pandas as pd

# ── Ścieżki ───────────────────────────────────────────────────────────────────
INPUT_PATH  = "dataset/dataset.csv"
OUTPUT_PATH = "dataset/data_no_outliers.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Próg oparty na: Faria et al. (2024), "Challenges in the care and treatment
# of patients with extreme obesity", World Journal of Gastroenterology.
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11326745/
# BMI >= 50 = "extreme obesity" (klasa IV/V), dotyczy ~0.5% populacji USA.
BMI_UPPER = 50.0
BMI_LOWER = 10.0 

df = pd.read_csv(INPUT_PATH)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

mask = (df["bmi"] < BMI_LOWER) | (df["bmi"] > BMI_UPPER)

for idx in df[mask].index:
    print(f"usuwam wiersz id={df.loc[idx, 'id']}  |  bmi={df.loc[idx, 'bmi']}")

df_clean = df[~mask]

print(f"\nWierszy przed: {len(df)}  |  usuniętych: {mask.sum()}  |  po: {len(df_clean)}")
df_clean.to_csv(OUTPUT_PATH, index=False)