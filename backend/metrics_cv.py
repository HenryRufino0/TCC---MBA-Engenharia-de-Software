from dataclasses import asdict, dataclass
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class FoldMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def _youden(tpr: np.ndarray, fpr: np.ndarray, thr: np.ndarray) -> Tuple[float, float, float]:
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(tpr[i]), float(fpr[i]), float(thr[i])


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000,
                     rng: int = 42) -> Tuple[float, float]:
    """IC 95% por bootstrap estratificado da AUC."""
    rng = np.random.RandomState(rng)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n_boot):
    
        s_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        s_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        s = np.concatenate([s_pos, s_neg])
        aucs.append(roc_auc_score(y_true[s], y_score[s]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def crossval_roc_calibrated(estimator, X, y, k: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    CV estratificada k-fold com calibração isotônica por fold.
    Retorna ROC média (com banda), métricas média±dp e AUC com IC 95% (bootstrap OOF).
    """
    X = np.asarray(X) if not hasattr(X, "iloc") else X  
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    mean_fpr = np.linspace(0, 1, 101)
    tprs: List[np.ndarray] = []
    aucs: List[float] = []
    fold_metrics: List[FoldMetrics] = []

    y_true_oof = np.zeros_like(y, dtype=int)
    y_score_oof = np.zeros_like(y, dtype=float)

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        base = clone(estimator)

        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
        calibrated.fit(X[tr] if isinstance(X, np.ndarray) else X.iloc[tr], y[tr])

        prob = calibrated.predict_proba(X[te] if isinstance(X, np.ndarray) else X.iloc[te])[:, 1]
        fpr_k, tpr_k, thr_k = roc_curve(y[te], prob)
        auc_k = roc_auc_score(y[te], prob)
        aucs.append(auc_k)


        tpr_interp = np.interp(mean_fpr, fpr_k, tpr_k)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

   
        pred = (prob >= 0.5).astype(int)
        fold_metrics.append(FoldMetrics(
            accuracy_score(y[te], pred),
            precision_score(y[te], pred, zero_division=0),
            recall_score(y[te], pred, zero_division=0),
            f1_score(y[te], pred, zero_division=0),
            auc_k
        ))

   
        y_true_oof[te] = y[te]
        y_score_oof[te] = prob

 
    arr = np.array([[m.accuracy, m.precision, m.recall, m.f1, m.auc] for m in fold_metrics])
    avg = FoldMetrics(*np.mean(arr, axis=0))
    std = FoldMetrics(*np.std(arr, axis=0))

    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))


    ci_lo, ci_hi = bootstrap_auc_ci(y_true_oof, y_score_oof)

 
    fpr_oof, tpr_oof, thr_oof = roc_curve(y_true_oof, y_score_oof)
    tpr_star, fpr_star, tau_star = _youden(tpr_oof, fpr_oof, thr_oof)

    return {
        "k": k,
        "avg": asdict(avg),
        "std": asdict(std),
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_ci95": [ci_lo, ci_hi],
        "roc_mean": {"fpr": mean_fpr.tolist(), "tpr": mean_tpr.tolist()},
        "youden_oof": {"tpr": tpr_star, "fpr": fpr_star, "tau": tau_star},
        "oof": {  
            "fpr": fpr_oof.tolist(),
            "tpr": tpr_oof.tolist(),
            "thresholds": thr_oof.tolist()
        }
    }


def roc_holdout_calibrated(estimator, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Treina com calibração no treino e calcula ROC/Youden no holdout.
    """
    base = clone(estimator)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)


    j = tpr - fpr
    i = int(np.argmax(j))
    tau = float(thr[i])

 
    y_hat = (y_prob >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
        "tau_youden_holdout": tau,
        "sens_at_tau": float(sens),
        "spec_at_tau": float(spec)
    }
