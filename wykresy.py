import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ścieżki do plików
INPUT_FILE = "results/stat_tests.txt"
OUTPUT_DIR = "results/plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. PARSOWANIE PLIKU TEKSTOWEGO
# ==============================================================================
print("Parsowanie pliku z wynikami...")

metrics_data = []
wilcoxon_data = []

current_model = None
current_metric = None

with open(INPUT_FILE, "r") as f:
    for line in f:
        line = line.strip()

        # Szukanie modeli i ich metryk
        if line.startswith("Model:"):
            current_model = line.split(":")[1].strip()
        elif current_model and ":" in line and ("median=" in line or "mean=" in line):
            metric_name, values = line.split(":")
            metric_name = metric_name.strip()
            
            median_match = re.search(r"median=([0-9.eE+-]+)", values)
            mean_match = re.search(r"mean=([0-9.eE+-]+)", values)
            
            if median_match and mean_match:
                median_val = float(median_match.group(1))
                mean_val = float(mean_match.group(1))
                metrics_data.append(
                    {
                        "Model": current_model,
                        "Metryka": metric_name,
                        "Mediana": median_val,
                        "Średnia": mean_val,
                    }
                )

        # Szukanie sekcji Wilcoxona
        if line.startswith("Metric:"):
            current_metric = line.split(":")[1].strip()
            current_model = None  # Koniec sekcji metryk modeli
        elif current_metric and "vs" in line and "p=" in line:
            models_part, stats_part = line.split(":")
            m1, m2 = [m.strip() for m in models_part.split("vs")]

            p_match = re.search(r"p=([0-9.eE+-]+)", stats_part)
            diff_match = re.search(r"median_diff=([0-9.eE+-]+)", stats_part)

            if p_match and diff_match:
                p_val = float(p_match.group(1))
                diff_val = float(diff_match.group(1))
                wilcoxon_data.append(
                    {
                        "Metryka": current_metric,
                        "Model_1": m1,
                        "Model_2": m2,
                        "p-value": p_val,
                        "Różnica": diff_val,
                    }
                )

# Tworzenie obiektów DataFrame
df_metrics = pd.DataFrame(metrics_data)
df_wilcoxon = pd.DataFrame(wilcoxon_data)

# Zabezpieczenie na wypadek pustych danych
if df_metrics.empty or df_wilcoxon.empty:
    print("Błąd: Nie udało się sparsować żadnych danych. Sprawdź format pliku tekstowego.")
    exit()

# ==============================================================================
# 2. GENEROWANIE TABELI ZBIORCZEJ (HTML)
# ==============================================================================
df_pivot = df_metrics.pivot(
    index="Model", columns="Metryka", values="Mediana"
).round(4)
df_pivot = df_pivot[["accuracy", "precision", "recall", "f1", "roc_auc"]]

df_pivot.to_html(os.path.join(OUTPUT_DIR, "tabela_metryk.html"))
print(f"✓ Zapisano tabelę HTML do: {OUTPUT_DIR}tabela_metryk.html")

# ==============================================================================
# 3. WYKRES 1: PORÓWNANIE METRYK (BAR PLOT)
# ==============================================================================
print("Generowanie wykresu porównawczego metryk...")
plt.figure(figsize=(12, 7))

models_list = df_pivot.index.tolist()
metrics_list = df_pivot.columns.tolist()

x = np.arange(len(metrics_list))
width = 0.2
colors = ["#2b5c8f", "#4682b4", "#6baed6", "#b3cde3"]

for i, model in enumerate(models_list):
    values = df_pivot.loc[model].values
    plt.bar(
        x + (i * width) - (width * len(models_list) / 4),
        values,
        width,
        label=model,
        color=colors[i],
        edgecolor="black",
        alpha=0.9,
    )

plt.ylabel("Wartość (Mediana z 30 powtórzeń)", fontsize=12, fontweight="bold")
plt.title(
    "Porównanie wydajności modeli (Dane bez SMOTE)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xticks(x + width / 4, [m.upper() for m in metrics_list], fontsize=11)
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "porownanie_modeli.png"), dpi=300)
plt.close()

# ==============================================================================
# 4. WYKRES 2: HEATMAPY ISTOTNOŚCI STATYSTYCZNEJ (WILCOXON)
# ==============================================================================
print("Generowanie heatmap testów statystycznych...")


def generate_heatmap(metric_name, filename, title):
    df_m = df_wilcoxon[df_wilcoxon["Metryka"] == metric_name]
    if df_m.empty:
        return

    all_models = sorted(
        list(set(df_m["Model_1"].tolist() + df_m["Model_2"].tolist()))
    )
    matrix_p = pd.DataFrame(1.0, index=all_models, columns=all_models)

    for _, row in df_m.iterrows():
        m1, m2, p = row["Model_1"], row["Model_2"], row["p-value"]
        matrix_p.loc[m1, m2] = p
        matrix_p.loc[m2, m1] = p

    plt.figure(figsize=(8, 6))
    
    # Rysujemy heatmapę z jawną skalą do 0.05
    sns.heatmap(
        matrix_p,
        annot=True,
        fmt=".3e",
        cmap="RdYlGn_r",
        cbar=True,
        linewidths=1,
        linecolor="white",
        vmax=0.05,
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


generate_heatmap(
    "f1",
    "wilcoxon_heatmap_f1.png",
    "Istotność statystyczna różnic dla metryki F1-Score\n(Zielony = Różnica istotna statystycznie, p < 0.05)",
)

generate_heatmap(
    "roc_auc",
    "wilcoxon_heatmap_auc.png",
    "Istotność statystyczna różnic dla metryki AUC-ROC\n(Zielony = Różnica istotna statystycznie, p < 0.05)",
)

print(f"✓ Wszystkie wykresy zostały pomyślnie zapisane w folderze: {OUTPUT_DIR}")