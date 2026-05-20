import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATASET_PATH     = "dataset/data_no_outliers.csv"
PREPROCESSED_PATH = "dataset/dataset_prep_PCA.csv"

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
    df["ever_married"]  = df["ever_married"].map({"Yes": 1, "No": 0})
    df["gender"]        = df["gender"].map({"Male": 1, "Female": 0})

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["Residence_type", "work_type", "smoking_status"],
                        drop_first=False, dtype=int)

    # Standaryzacja zmiennych numerycznych (wymagana przez PCA)
    num_cols = ["age", "avg_glucose_level", "bmi"]
    std_scaler = StandardScaler()
    df[num_cols] = std_scaler.fit_transform(df[num_cols])

    # Uzupełnienie brakujących wartości medianą (np. ever_married po mapowaniu)
    df.fillna(df.median(numeric_only=True), inplace=True)

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

# ── PCA ───────────────────────────────────────────────────────────────────────

pca = PCA()
components = pca.fit_transform(df)

explained_variance   = pca.explained_variance_ratio_
cumulative_variance  = np.cumsum(explained_variance)

print("\nPCA — wariancja wyjaśniona przez kolejne składowe:")
print(f"{'PC':<6} {'Wariancja':>12} {'Skumulowana':>14}")
print("─" * 34)
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance), 1):
    print(f"PC{i:<4} {ev:>11.4f}  {cv:>13.4f}")

n_95 = next(i for i, cv in enumerate(cumulative_variance, 1) if cv >= 0.95)
print(f"\nSkładowe potrzebne do wyjaśnienia 95% wariancji: {n_95}")

# ── Wizualizacje ──────────────────────────────────────────────────────────────

COLORS = {0: "#4A90D9", 1: "#E84040"}
LABELS = {0: "Brak udaru", 1: "Udar"}

# Wykres: Explained variance
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.bar(range(1, len(explained_variance) + 1), explained_variance,
        color="#4A90D9", alpha=0.7, label="Wariancja składowej")
ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
         color="#E84040", marker="o", linewidth=2, markersize=4, label="Skumulowana")
ax1.axhline(y=0.95, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax1.text(len(explained_variance) * 0.6, 0.96, "95%", color="gray", fontsize=9)
ax1.set_xlabel("Składowa główna (PC)")
ax1.set_ylabel("Odsetek wyjaśnionej wariancji")
ax1.set_title("PCA — Wyjaśniona wariancja")
ax1.set_xticks(range(1, len(explained_variance) + 1))
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
fig1.tight_layout()
fig1.savefig("saved_plots/pca_explained_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano wykres: pca_explained_variance.png")

# Scatter plot PC1 vs PC2 (2D)
fig2, ax2 = plt.subplots(figsize=(8, 6))
for label, color in COLORS.items():
    mask = stroke == label
    ax2.scatter(
        components[mask, 0], components[mask, 1],
        c=color, label=LABELS[label],
        alpha=0.5 if label == 0 else 0.9,
        s=15 if label == 0 else 25,
        edgecolors="none"
    )
ax2.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% wariancji)")
ax2.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% wariancji)")
ax2.set_title("PCA — Scatter plot PC1 vs PC2")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("saved_plots/pca_scatter_2d.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano wykres: pca_scatter_2d.png")

# Scatter plot PC1 vs PC2 vs PC3 (3D)
fig3 = plt.figure(figsize=(9, 7))
ax3 = fig3.add_subplot(111, projection="3d")
for label, color in COLORS.items():
    mask = stroke == label
    ax3.scatter(
        components[mask, 0], components[mask, 1], components[mask, 2],
        c=color, label=LABELS[label],
        alpha=0.4 if label == 0 else 0.8,
        s=10 if label == 0 else 20,
        edgecolors="none"
    )
ax3.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
ax3.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
ax3.set_zlabel(f"PC3 ({explained_variance[2]*100:.1f}%)")
ax3.set_title("PCA — Scatter plot PC1 vs PC2 vs PC3")
ax3.legend()
fig3.tight_layout()
fig3.savefig("saved_plots/pca_scatter_3d.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano wykres: pca_scatter_3d.png")