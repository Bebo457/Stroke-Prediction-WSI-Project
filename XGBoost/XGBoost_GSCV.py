import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

HERE         = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(HERE)
DATASET_PATH = os.path.join(ROOT, "dataset", "dataset_prep.csv")
RESULTS_DIR  = os.path.join(HERE, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dane ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATASET_PATH)
X  = df.drop(columns=["stroke"])
y  = df["stroke"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Val: {X_val.shape}\n")

# ── Pipeline: SMOTE → PCA → XGBoost ──────────────────────────────────────────
# SMOTE i PCA są wewnątrz pipeline'u, więc w każdym fodzie CV są fitowane tylko
# na danych treningowych — bez wycieku do foldów walidacyjnych.

pipeline = Pipeline([
    ("smote",      SMOTE(random_state=42)),
    ("pca",        PCA(n_components=0.95, random_state=42)),
    ("classifier", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)),
])

# ── Siatka hiperparametrów ────────────────────────────────────────────────────

# 4×3×3×2×3×3 = 648 kombinacji × 5 foldów = 3240 fitów (~10 min przy 8 rdzeniach)
param_grid = {
    "classifier__max_depth":        [3, 4, 5, 6],
    "classifier__learning_rate":    [0.01, 0.05, 0.1],
    "classifier__n_estimators":     [300, 600, 1000],
    "classifier__min_child_weight": [1, 3],
    "classifier__gamma":            [0, 0.1, 0.3],
    "classifier__reg_lambda":       [1, 5, 10],
}

# ── Grid Search CV ────────────────────────────────────────────────────────────

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=3,
    refit=True,
)

print("Rozpoczynam Grid Search CV...\n")
grid_search.fit(X_train, y_train)

# ── Wyniki ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("NAJLEPSZE HIPERPARAMETRY:")
for param, value in grid_search.best_params_.items():
    print(f"  {param.replace('classifier__', '')}: {value}")
print(f"\nNajlepszy AUC-ROC (CV): {grid_search.best_score_:.4f}")
print("=" * 60)

output_path = os.path.join(RESULTS_DIR, "GSCV_parameters.txt")
with open(output_path, "w") as f:
    f.write("NAJLEPSZE HIPERPARAMETRY (GridSearchCV – SMOTE + PCA, zbiór treningowy)\n")
    f.write("=" * 60 + "\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param.replace('classifier__', '')}: {value}\n")
    f.write(f"\nNajlepszy AUC-ROC (CV): {grid_search.best_score_:.4f}\n")

print(f"\nZapisano: {output_path}")
