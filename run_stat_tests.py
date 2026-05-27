import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone
from scipy.stats import wilcoxon, chi2


def load_data(base_dir="dataset"):
    train = pd.read_csv(os.path.join(base_dir, "train.csv"))
    val = pd.read_csv(os.path.join(base_dir, "validation.csv"))
    test = pd.read_csv(os.path.join(base_dir, "test.csv"))
    return train, val, test


def split_X_y(df, target_col="stroke"):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y


def get_models():
    models = {}
    # POPRAWKA: Dodajemy balansowanie klas, bo zbiory walidacyjne i testowe mają tylko 4% pozytywów
    models['LogisticRegression'] = LogisticRegression(solver='liblinear', class_weight='balanced')
    models['RandomForest'] = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    models['SVM'] = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    try:
        from xgboost import XGBClassifier
        # POPRAWKA: Dla XGBoost dodajemy scale_pos_weight=24 dla zrównoważenia asymetrii
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=24, random_state=42)
    except Exception:
        pass
    return models


def compute_metrics(y_true, y_pred, y_proba=None):
    out = {}
    out['accuracy'] = accuracy_score(y_true, y_pred)
    out['precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['recall'] = recall_score(y_true, y_pred, zero_division=0)
    out['f1'] = f1_score(y_true, y_pred, zero_division=0)
    if y_proba is not None:
        try:
            out['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception:
            out['roc_auc'] = np.nan
    else:
        out['roc_auc'] = np.nan
    return out


def repeated_evaluation(models, X_train, y_train, X_val, y_val, n_repeats=30, random_seed=0):
    rng = np.random.RandomState(random_seed)
    results = {name: defaultdict(list) for name in models}

    for i in range(n_repeats):
        rs = int(rng.randint(0, 2 ** 31 - 1))
        for name, model in models.items():
            m = clone(model)
            if hasattr(m, 'random_state') and name != 'XGBoost':
                try:
                    m.random_state = rs
                except Exception:
                    pass
            m.fit(X_train, y_train)
            y_pred = m.predict(X_val)
            y_proba = None
            if hasattr(m, 'predict_proba'):
                try:
                    y_proba = m.predict_proba(X_val)[:, 1]
                except Exception:
                    y_proba = None
            elif hasattr(m, 'decision_function'):
                try:
                    dec = m.decision_function(X_val)
                    y_proba = 1.0 / (1.0 + np.exp(-dec))
                except Exception:
                    y_proba = None

            metrics = compute_metrics(y_val, y_pred, y_proba)
            for k, v in metrics.items():
                results[name][k].append(v)

    return results


def pairwise_wilcoxon(results, metric):
    names = list(results.keys())
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(results[names[i]][metric])
            b = np.array(results[names[j]][metric])
            try:
                stat, p = wilcoxon(a, b)
            except Exception:
                stat, p = np.nan, np.nan
            out.append((names[i], names[j], stat, p, np.median(a - b)))
    return out


# POPRAWKA: Naprawiona matematyka testu McNemara z uwzględnieniem y_true
def mcnemar_test(y_true, y_pred1, y_pred2):
    assert len(y_true) == len(y_pred1) == len(y_pred2)
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))  # Model 1 ma rację, Model 2 się myli
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))  # Model 1 się myli, Model 2 ma rację
    
    denom = b + c
    if denom == 0:
        return np.nan, np.nan
    stat = (abs(b - c) - 1) ** 2 / denom
    pvalue = chi2.sf(stat, df=1)
    return stat, pvalue


# POPRAWKA: Przekazujemy teraz y_test do funkcji testującej McNemara
def pairwise_mcnemar_once(models, X_train, y_train, X_test, y_test):
    preds = {}
    for name, model in models.items():
        m = clone(model)
        m.fit(X_train, y_train)
        preds[name] = m.predict(X_test)

    names = list(preds.keys())
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            stat, p = mcnemar_test(y_test, preds[names[i]], preds[names[j]])
            out.append((names[i], names[j], stat, p))
    return out


def save_results_text(path, repeated_res, wilcoxon_res_by_metric, mcnemar_res):
    with open(path, 'w') as f:
        f.write('Repeated evaluation summary (medians):\n')
        for name, metrics in repeated_res.items():
            f.write(f"Model: {name}\n")
            for k, vals in metrics.items():
                f.write(f"  {k}: median={np.nanmedian(vals):.4f} mean={np.nanmean(vals):.4f}\n")
        f.write('\nWilcoxon pairwise tests:\n')
        for metric, rows in wilcoxon_res_by_metric.items():
            f.write(f"Metric: {metric}\n")
            for a, b, stat, p, meddiff in rows:
                f.write(f"  {a} vs {b}: stat={stat}, p={p}, median_diff={meddiff:.4f}\n")
        f.write('\nMcNemar tests on single train->test fit:\n')
        for a, b, stat, p in mcnemar_res:
            f.write(f"  {a} vs {b}: stat={stat}, p={p}\n")


def main():
    parser = argparse.ArgumentParser(description='Run statistical tests comparing basic classifiers')
    parser.add_argument('--dataset-dir', default='dataset')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='results/stat_tests.txt')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train_df, val_df, test_df = load_data(args.dataset_dir)
    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)
    X_test, y_test = split_X_y(test_df)

    models = get_models()
    print('Models to evaluate:', list(models.keys()))

    repeated_res = repeated_evaluation(models, X_train, y_train, X_val, y_val, n_repeats=args.repeats, random_seed=args.seed)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    wilcoxon_res_by_metric = {}
    for m in metrics:
        wilcoxon_res_by_metric[m] = pairwise_wilcoxon(repeated_res, m)

    mcnemar_res = pairwise_mcnemar_once(models, X_train, y_train, X_test, y_test)

    save_results_text(args.out, repeated_res, wilcoxon_res_by_metric, mcnemar_res)
    print('Results saved to', args.out)


if __name__ == '__main__':
    main()