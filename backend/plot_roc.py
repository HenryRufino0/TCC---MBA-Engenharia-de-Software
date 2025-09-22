import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.base import clone


CANDIDATES = [
    os.environ.get("DATASET_PATH"),
    "data/automation_data.csv",
    "../data/automation_data.csv",
    "automation_data.csv",
]
csv_path = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(
        "Não achei o dataset. Defina DATASET_PATH ou coloque 'automation_data.csv' em ./data/ ou ../data/."
    )


df = pd.read_csv(csv_path)


if "automatizar" not in df.columns:
    raise ValueError("Coluna 'automatizar' não encontrada no CSV.")

df["automatizar"] = (
    df["automatizar"].astype(str).str.strip().str.lower()
      .map({"sim": 1, "nao": 0, "não": 0, "0": 0, "1": 1})
)
if df["automatizar"].isna().any():
    bad = df.loc[df["automatizar"].isna(), "automatizar"]
    raise ValueError(f"Valores inválidos em 'automatizar'. Ex.: {bad.unique()[:5]}")
y = df["automatizar"].astype(int).values


drop_cols = {"automatizar", "nome_tarefa", "id", "data", "data_registro"}
Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns])


if "complexidade" in Xdf.columns:
    Xdf["complexidade"] = (
        Xdf["complexidade"].astype(str).str.strip().str.lower()
        .map({"baixa": 1, "media": 2, "média": 2, "alta": 3})
        .fillna(2).astype(int)
    )

for col in Xdf.select_dtypes(include=["object"]).columns:
    Xdf[col] = Xdf[col].astype("category").cat.codes

X = Xdf.values


X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_tr, y_tr)


y_prob = model.predict_proba(X_te)[:, 1]
fpr, tpr, thr = roc_curve(y_te, y_prob)
auc = roc_auc_score(y_te, y_prob)
youden = tpr - fpr
ix = int(np.argmax(youden))
tau = float(thr[ix])  

with open("roc_holdout.json", "w", encoding="utf-8") as f:
    json.dump(
        {"auc": float(auc), "fpr": fpr.tolist(), "tpr": tpr.tolist(),
         "thresholds": thr.tolist(), "tau_youden": tau},
        f, ensure_ascii=False, indent=2
    )

plt.figure()
plt.plot(fpr, tpr, label=f"Holdout ROC (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Aleatório")
plt.scatter([fpr[ix]], [tpr[ix]], label=f"Youden τ={tau:.3f}", s=30)
plt.xlabel("FPR (1-Especificidade)"); plt.ylabel("TPR (Sensibilidade)")
plt.title("Curva ROC — Holdout"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("roc_holdout.png", dpi=200, bbox_inches="tight")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_fpr = np.linspace(0, 1, 101)
tprs, aucs, metrics = [], [], []

for tr_idx, te_idx in skf.split(X, y):
    clf = clone(model).fit(X[tr_idx], y[tr_idx])
    prob = clf.predict_proba(X[te_idx])[:, 1]
    pred = (prob >= tau).astype(int)  

    aucs.append(roc_auc_score(y[te_idx], prob))
    fpr_k, tpr_k, _ = roc_curve(y[te_idx], prob)
    tprs.append(np.interp(mean_fpr, fpr_k, tpr_k))

    metrics.append({
        "accuracy": float(accuracy_score(y[te_idx], pred)),
        "precision": float(precision_score(y[te_idx], pred, zero_division=0)),
        "recall": float(recall_score(y[te_idx], pred, zero_division=0)),
        "f1": float(f1_score(y[te_idx], pred, zero_division=0)),
    })

mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
cv_summary = {
    "k": 5,
    "auc_mean": float(np.mean(aucs)),
    "auc_std": float(np.std(aucs)),
    "metrics_mean": {m: float(np.mean([d[m] for d in metrics])) for m in metrics[0]},
    "metrics_std": {m: float(np.std([d[m] for d in metrics])) for m in metrics[0]},
    "roc_mean": {"fpr": mean_fpr.tolist(), "tpr": mean_tpr.tolist()}
}
with open("metricas_cv.json", "w", encoding="utf-8") as f:
    json.dump(cv_summary, f, ensure_ascii=False, indent=2)

plt.figure()
plt.plot(mean_fpr, mean_tpr, label=f"ROC média (AUC={cv_summary['auc_mean']:.3f}±{cv_summary['auc_std']:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Aleatório")
plt.xlabel("FPR (1-Especificidade)"); plt.ylabel("TPR (Sensibilidade)")
plt.title("Curva ROC — Média 5-Fold"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("roc_cv_media.png", dpi=200, bbox_inches="tight")

print("OK: gerados 'roc_holdout.png', 'roc_cv_media.png', 'roc_holdout.json' e 'metricas_cv.json'")
