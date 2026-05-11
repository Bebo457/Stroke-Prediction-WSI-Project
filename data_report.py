import pandas as pd
import numpy as np

# ── Wczytanie danych ──────────────────────────────────────────────────────────
df = pd.read_csv("dataset/dataset.csv")

# zamień "N/A" (string) na prawdziwy NaN
df.replace("N/A", np.nan, inplace=True)

SEP = "─" * 60

# ── 1. Podstawowe info ────────────────────────────────────────────────────────
print(SEP)
print(f"  LICZBA REKORDÓW:  {len(df)}")
print(f"  LICZBA KOLUMN:    {len(df.columns)}")
print(SEP)

# ── 2. Typy danych i brakujące wartości ───────────────────────────────────────
print("\n📋  KOLUMNY — TYP / BRAKUJĄCE WARTOŚCI\n")
print(f"{'Kolumna':<22} {'Typ':<12} {'Braków':>8} {'% braków':>10}")
print("─" * 56)
for col in df.columns:
    missing = df[col].isna().sum()
    pct     = missing / len(df) * 100
    print(f"{col:<22} {str(df[col].dtype):<12} {missing:>8} {pct:>9.2f}%")

# ── 3. Zmienne kategoryczne — rozkład wartości ────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()

print(f"\n\n{'='*60}")
print("  ZMIENNE KATEGORYCZNE — ROZKŁAD WARTOŚCI")
print(f"{'='*60}")

for col in cat_cols:
    counts = df[col].value_counts(dropna=False)
    print(f"\n▸ {col}")
    print(f"  {'Wartość':<25} {'Liczba':>8} {'%':>8}")
    print("  " + "─" * 44)
    for val, cnt in counts.items():
        pct = cnt / len(df) * 100
        label = str(val) if pd.notna(val) else "NaN"
        print(f"  {label:<25} {cnt:>8} {pct:>7.2f}%")

# ── 4. Zmienne numeryczne — statystyki ────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n\n{'='*60}")
print("  ZMIENNE NUMERYCZNE — STATYSTYKI")
print(f"{'='*60}")

for col in num_cols:
    s = df[col].dropna()
    print(f"\n▸ {col}  (n={len(s)}, braków={df[col].isna().sum()})")
    print(f"  min={s.min():.2f}  max={s.max():.2f}  "
          f"mean={s.mean():.2f}  median={s.median():.2f}  std={s.std():.2f}")

    # histogram tekstowy (10 kubełków)
    counts, edges = np.histogram(s, bins=10)
    max_bar = 30
    scale   = max_bar / max(counts) if max(counts) > 0 else 1
    print()
    for i, c in enumerate(counts):
        bar   = "█" * int(c * scale)
        label = f"  [{edges[i]:7.1f} – {edges[i+1]:7.1f}]"
        print(f"{label}  {bar}  ({c})")

# ── 5. Zmienna docelowa ───────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("  ZMIENNA DOCELOWA — stroke")
print(f"{'='*60}")
vc = df["stroke"].value_counts()
for val, cnt in vc.items():
    label = "udar     (1)" if val == 1 else "brak udaru (0)"
    print(f"  {label}: {cnt:>5}  ({cnt/len(df)*100:.2f}%)")
print(f"\n  Stosunek klas (0:1) = {vc[0]/vc[1]:.1f}:1  ← silne niezbalansowanie!")

print(f"\n{SEP}\n  Gotowe.\n{SEP}\n")