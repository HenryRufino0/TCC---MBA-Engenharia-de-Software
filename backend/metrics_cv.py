import numpy as np
from dataclasses import dataclass, asdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.base import clone

@dataclass
class FoldMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float

def crossval_cart(model, X, y, k=5, random_state=42):
    """
    Executa validação cruzada estratificada (k-fold) e retorna
    métricas média±dp e a ROC média.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    metrics = []
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []

    for train_idx, test_idx in skf.split(X, y):
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y[train_idx])

        y_prob = model_clone.predict_proba(X[test_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y[test_idx], y_pred)
        prec = precision_score(y[test_idx], y_pred, zero_division=0)
        rec = recall_score(y[test_idx], y_pred, zero_division=0)
        f1 = f1_score(y[test_idx], y_pred, zero_division=0)
        auc = roc_auc_score(y[test_idx], y_prob)

        fpr, tpr, _ = roc_curve(y[test_idx], y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        metrics.append(FoldMetrics(acc, prec, rec, f1, auc))

    arr = np.array([[m.accuracy, m.precision, m.recall, m.f1, m.auc] for m in metrics])
    avg = FoldMetrics(*np.mean(arr, axis=0))
    std = FoldMetrics(*np.std(arr, axis=0))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    return {
        "k": k,
        "avg": asdict(avg),
        "std": asdict(std),
        "roc_mean": {"fpr": mean_fpr.tolist(), "tpr": mean_tpr.tolist()}
    }

def roc_holdout(model, X_test, y_test):
    """
    Calcula a ROC e o AUC no conjunto de teste (holdout).
    """
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return {
        "auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist()
    }
