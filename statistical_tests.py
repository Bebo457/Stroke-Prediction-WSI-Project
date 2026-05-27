"""
STATYSTYCZNE PORÓWNANIE MODELI KLASYFIKACJI
Logistic Regression vs SVM vs XGBoost vs Random Forest
Na podstawowych wersjach algorytmów i danych: train.csv, test.csv, validation.csv

Przeprowadzane testy:
1. Metryki podstawowe (Accuracy, Precision, Recall, F1, AUC-ROC)
2. Macierze konfuzji
3. Walidacja krzyżowa (5-fold cross-validation)
4. Test McNemara (porównanie parami)
5. Przedziały ufności (95%)
6. Analiza zmienności wyników
7. Wizualizacje porównawcze
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# =========================
# KONFIGURACJA
# =========================

RANDOM_STATE = 42
CV_SPLITS = 5
CONFIDENCE_LEVEL = 0.95

# =========================
# WCZYTANIE DANYCH
# =========================

print("=" * 80)
print("ŁADOWANIE DANYCH")
print("=" * 80)

train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

X_train = train_df.drop("stroke", axis=1)
y_train = train_df["stroke"]

X_val = val_df.drop("stroke", axis=1)
y_val = val_df["stroke"]

X_test = test_df.drop("stroke", axis=1)
y_test = test_df["stroke"]

print(f"\nRozmiary zbiorów:")
print(f"  Train: {X_train.shape} (udary: {y_train.sum()} - {y_train.mean()*100:.2f}%)")
print(f"  Val:   {X_val.shape}   (udary: {y_val.sum()} - {y_val.mean()*100:.2f}%)")
print(f"  Test:  {X_test.shape}  (udary: {y_test.sum()} - {y_test.mean()*100:.2f}%)")

# =========================
# INICJALIZACJA MODELI
# =========================

print("\n" + "=" * 80)
print("INICJALIZACJA MODELI")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=RANDOM_STATE
    ),
    'SVM': SVC(
        kernel='rbf', 
        C=1, 
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_STATE
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

print("\nMemodele zainicjalizowane:")
for name in models.keys():
    print(f"  ✓ {name}")

# =========================
# TRENOWANIE MODELI
# =========================

print("\n" + "=" * 80)
print("TRENOWANIE MODELI")
print("=" * 80)

trained_models = {}
for name, model in models.items():
    print(f"\nTrenowanie {name}...", end=" ")
    if name == 'XGBoost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    trained_models[name] = model
    print("✓")

print("\nWszystkie modele wytrenowane!")

# =========================
# WALIDACJA NA ZBIORZE TESTOWYM
# =========================

print("\n" + "=" * 80)
print("WALIDACJA NA ZBIORZE TESTOWYM")
print("=" * 80)

test_results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    
    test_results[name] = {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'cm': confusion_matrix(y_test, y_pred)
    }

# Tabelaryczne wyniki testu
print("\n" + "─" * 80)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}")
print("─" * 80)
for name, results in test_results.items():
    print(f"{name:<25} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
          f"{results['recall']:<12.4f} {results['f1']:<12.4f} {results['auc_roc']:<12.4f}")
print("─" * 80)

# =========================
# WALIDACJA KRZYŻOWA (5-FOLD)
# =========================

print("\n" + "=" * 80)
print("WALIDACJA KRZYŻOWA (5-fold stratified)")
print("=" * 80)

# Połączenie danych do CV
X_cv = pd.concat([X_train, X_val], ignore_index=True)
y_cv = pd.concat([y_train, y_val], ignore_index=True)

cv_results = {}
skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for name, model in trained_models.items():
    print(f"\n{name}:")
    
    scores = cross_validate(
        model, X_cv, y_cv,
        cv=skf,
        scoring=scoring_metrics,
        return_train_score=False
    )
    
    cv_results[name] = {}
    for metric in scoring_metrics:
        metric_scores = scores[f'test_{metric}']
        mean_score = metric_scores.mean()
        std_score = metric_scores.std()
        
        # Przedział ufności (95%)
        ci = stats.t.interval(
            CONFIDENCE_LEVEL,
            len(metric_scores) - 1,
            loc=mean_score,
            scale=stats.sem(metric_scores)
        )
        
        cv_results[name][metric] = {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'scores': metric_scores
        }
        
        print(f"  {metric.capitalize():<15}: {mean_score:.4f} ± {std_score:.4f} "
              f"[95% CI: {ci[0]:.4f} - {ci[1]:.4f}]")

# =========================
# MACIERZE KONFUZJI
# =========================

print("\n" + "=" * 80)
print("MACIERZE KONFUZJI (zbiór testowy)")
print("=" * 80)

for name, results in test_results.items():
    cm = results['cm']
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{name}:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    print(f"  Specificity:     {specificity:.4f}")

# =========================
# TEST McNEMARA (porównanie parami)
# =========================

print("\n" + "=" * 80)
print("TEST McNEMARA (porównanie parami modeli)")
print("=" * 80)

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Test McNemara do porównania dwóch klasyfikatorów.
    H0: Oba klasyfikatory mają tę samą liczbę błędów
    """
    # Obliczenie zgodności/niezgodności
    diff = y_pred1 != y_pred2
    
    # Gdzie pierwszy jest dobry, drugi źle
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    
    # Gdzie pierwszy jest źle, drugi dobry
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    # Test statystyczny
    if b + c > 0:
        chi2 = ((b - c) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0
        p_value = 1.0
    
    return chi2, p_value, b, c

model_names = list(test_results.keys())
n_models = len(model_names)

print("\nWyniku testów McNemara (p-wartości):")
print("(H0: modele mają równą wydajność)")
print()

mcnemar_matrix = np.zeros((n_models, n_models))
significance_matrix = np.zeros((n_models, n_models), dtype=object)

for i in range(n_models):
    for j in range(i+1, n_models):
        name1, name2 = model_names[i], model_names[j]
        y_pred1 = test_results[name1]['y_pred']
        y_pred2 = test_results[name2]['y_pred']
        
        chi2, p_value, b, c = mcnemar_test(y_test, y_pred1, y_pred2)
        
        mcnemar_matrix[i, j] = p_value
        mcnemar_matrix[j, i] = p_value
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{name1:25} vs {name2:25}: "
              f"χ² = {chi2:7.4f}, p = {p_value:.6f} {significance}")
        print(f"  (Model {name1:20} wygrał {b} razy, {name2:20} wygrał {c} razy)")

# =========================
# PRZEDZIAŁY UFNOŚCI DLA METRYK TESTOWYCH
# =========================

print("\n" + "=" * 80)
print("PRZEDZIAŁY UFNOŚCI (95%) - ZBIÓR TESTOWY")
print("=" * 80)

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence intervals"""
    scores = []
    n = len(y_true)
    
    np.random.seed(RANDOM_STATE)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        scores.append(metric_func(y_true[indices], y_pred[indices]))
    
    alpha = 1 - ci
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    
    return lower, upper, np.mean(scores)

print()
for name, results in test_results.items():
    y_pred = results['y_pred']
    
    # Bootstrap dla każdej metryki
    acc_ci = bootstrap_ci(y_test.values, y_pred, accuracy_score)
    prec_ci = bootstrap_ci(y_test.values, y_pred, precision_score)
    rec_ci = bootstrap_ci(y_test.values, y_pred, recall_score)
    f1_ci = bootstrap_ci(y_test.values, y_pred, f1_score)
    
    print(f"{name}:")
    print(f"  Accuracy:  {acc_ci[2]:.4f} [95% CI: {acc_ci[0]:.4f} - {acc_ci[1]:.4f}]")
    print(f"  Precision: {prec_ci[2]:.4f} [95% CI: {prec_ci[0]:.4f} - {prec_ci[1]:.4f}]")
    print(f"  Recall:    {rec_ci[2]:.4f} [95% CI: {rec_ci[0]:.4f} - {rec_ci[1]:.4f}]")
    print(f"  F1-score:  {f1_ci[2]:.4f} [95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f}]")
    print()

# =========================
# RAPORTY KLASYFIKACJI
# =========================

print("\n" + "=" * 80)
print("RAPORTY KLASYFIKACJI (zbiór testowy)")
print("=" * 80)

for name, results in test_results.items():
    print(f"\n{name}")
    print("─" * 80)
    print(classification_report(y_test, results['y_pred'],
                               target_names=['Bez udaru', 'Udar']))

# =========================
# WIZUALIZACJE
# =========================

print("\n" + "=" * 80)
print("GENEROWANIE WIZUALIZACJI")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Porównanie metryk na zbiorze testowym
ax1 = axes[0, 0]
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
x = np.arange(len(metrics_to_plot))
width = 0.2

for idx, (name, results) in enumerate(test_results.items()):
    values = [results[metric] for metric in metrics_to_plot]
    ax1.bar(x + idx*width, values, width, label=name, alpha=0.8)

ax1.set_ylabel('Wartość', fontsize=12, fontweight='bold')
ax1.set_title('Porównanie metryk na zbiorze testowym', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels([m.upper() for m in metrics_to_plot], rotation=45)
ax1.legend()
ax1.grid(alpha=0.3, axis='y')
ax1.set_ylim([0, 1])

# 2. Krzywe ROC
ax2 = axes[0, 1]
for name, results in test_results.items():
    y_proba = results['y_proba']
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = results['auc_roc']
    ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax2.set_title('Krzywe ROC na zbiorze testowym', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

# 3. Dokładność w walidacji krzyżowej
ax3 = axes[1, 0]
cv_accuracies = []
cv_labels = []
for name, cv_info in cv_results.items():
    cv_accuracies.append(cv_info['accuracy']['scores'])
    cv_labels.append(name)

bp = ax3.boxplot(cv_accuracies, labels=cv_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']):
    patch.set_facecolor(color)

ax3.set_ylabel('Dokładność (Accuracy)', fontsize=12, fontweight='bold')
ax3.set_title('Dokładność w 5-fold CV', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# 4. Heatmapa wartości AUC-ROC
ax4 = axes[1, 1]
auc_values = np.array([[test_results[name]['auc_roc'] for name in model_names]]).T
im = ax4.imshow(auc_values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1)
ax4.set_yticks(range(len(model_names)))
ax4.set_yticklabels(model_names)
ax4.set_xticks([0])
ax4.set_xticklabels(['AUC-ROC'])
ax4.set_title('Wartości AUC-ROC na zbiorze testowym', fontsize=14, fontweight='bold')

for i, val in enumerate(auc_values.flatten()):
    ax4.text(0, i, f'{val:.4f}', ha='center', va='center', fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax4)
plt.tight_layout()
plt.savefig('statistical_tests_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Zapisano: statistical_tests_comparison.png")
plt.close()

# =========================
# MACIERZE KONFUZJI - WIZUALIZACJA
# =========================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, results) in enumerate(test_results.items()):
    cm = results['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Bez udaru', 'Udar'],
                yticklabels=['Bez udaru', 'Udar'],
                cbar_kws={'label': 'Liczba próbek'})
    axes[idx].set_title(f'Confusion Matrix - {name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Rzeczywista etykieta')
    axes[idx].set_xlabel('Przewidywana etykieta')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Zapisano: confusion_matrices_comparison.png")
plt.close()

# =========================
# WYKRESY WALIDACJI KRZYŻOWEJ
# =========================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    data_to_plot = []
    labels = []
    
    for name in model_names:
        data_to_plot.append(cv_results[name][metric]['scores'])
        labels.append(name)
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
    ax.set_title(f'{metric.upper()} w 5-fold CV', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.set_xticklabels(labels, rotation=45, ha='right')

# Usuń ostatni pusty subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('cv_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Zapisano: cv_metrics_comparison.png")
plt.close()

# =========================
# TABELARYCZNE PODSUMOWANIE CV
# =========================

print("\n" + "=" * 80)
print("PODSUMOWANIE WALIDACJI KRZYŻOWEJ")
print("=" * 80)

cv_summary = []
for name in model_names:
    row = {'Model': name}
    for metric in scoring_metrics:
        mean = cv_results[name][metric]['mean']
        std = cv_results[name][metric]['std']
        row[f'{metric.upper()} (mean ± std)'] = f"{mean:.4f} ± {std:.4f}"
    cv_summary.append(row)

cv_df = pd.DataFrame(cv_summary)
print("\n" + cv_df.to_string(index=False))

# =========================
# RANKING MODELI
# =========================

print("\n" + "=" * 80)
print("RANKING MODELI")
print("=" * 80)

metrics_weight = {
    'accuracy': 0.25,
    'precision': 0.25,
    'recall': 0.25,
    'f1': 0.25
}

ranking = {}
for name in model_names:
    score = 0
    for metric, weight in metrics_weight.items():
        score += test_results[name][metric] * weight
    ranking[name] = score

print("\nRanking na podstawie średniej ważonej metryk testowych:")
for rank, (name, score) in enumerate(sorted(ranking.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"  {rank}. {name:<25} - {score:.4f}")

# =========================
# ZAPISANIE RAPORTÓW
# =========================

print("\n" + "=" * 80)
print("ZAPISYWANIE RAPORTÓW")
print("=" * 80)

# Raport tekstowy
with open('statistical_tests_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RAPORT STATYSTYCZNEGO PORÓWNANIA MODELI KLASYFIKACJI\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("MODELE:\n")
    f.write("  1. Logistic Regression (klasa zbalansowana)\n")
    f.write("  2. Support Vector Machine (RBF kernel, klasa zbalansowana)\n")
    f.write("  3. XGBoost (100 drzew, depth=4)\n")
    f.write("  4. Random Forest (100 drzew, klasa zbalansowana)\n\n")
    
    f.write("DANE:\n")
    f.write(f"  Train: {X_train.shape}\n")
    f.write(f"  Validation: {X_val.shape}\n")
    f.write(f"  Test: {X_test.shape}\n\n")
    
    f.write("─" * 80 + "\n")
    f.write("WYNIKI NA ZBIORZE TESTOWYM\n")
    f.write("─" * 80 + "\n\n")
    
    for name, results in test_results.items():
        f.write(f"\n{name}\n")
        f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall:    {results['recall']:.4f}\n")
        f.write(f"  F1-score:  {results['f1']:.4f}\n")
        f.write(f"  AUC-ROC:   {results['auc_roc']:.4f}\n")
    
    f.write("\n" + "─" * 80 + "\n")
    f.write("WALIDACJA KRZYŻOWA (5-fold)\n")
    f.write("─" * 80 + "\n\n")
    
    for name in model_names:
        f.write(f"\n{name}\n")
        for metric in scoring_metrics:
            mean = cv_results[name][metric]['mean']
            std = cv_results[name][metric]['std']
            ci_lower = cv_results[name][metric]['ci_lower']
            ci_upper = cv_results[name][metric]['ci_upper']
            f.write(f"  {metric.upper():<12}: {mean:.4f} ± {std:.4f} ")
            f.write(f"[95% CI: {ci_lower:.4f} - {ci_upper:.4f}]\n")
    
    f.write("\n" + "─" * 80 + "\n")
    f.write("TEST McNEMARA\n")
    f.write("─" * 80 + "\n\n")
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            y_pred1 = test_results[name1]['y_pred']
            y_pred2 = test_results[name2]['y_pred']
            chi2, p_value, b, c = mcnemar_test(y_test, y_pred1, y_pred2)
            f.write(f"\n{name1} vs {name2}\n")
            f.write(f"  χ² = {chi2:.4f}, p-value = {p_value:.6f}\n")

print("✓ Zapisano: statistical_tests_report.txt")

# CSV z wynikami
results_df = pd.DataFrame(test_results).T
results_df.drop(['y_pred', 'y_proba', 'cm'], axis=1).to_csv('test_results.csv')
print("✓ Zapisano: test_results.csv")

cv_df.to_csv('cv_results.csv', index=False)
print("✓ Zapisano: cv_results.csv")

print("\n" + "=" * 80)
print("KONIEC")
print("=" * 80)
