import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from metrics_cv import crossval_roc_calibrated, roc_holdout_calibrated
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV



CANDIDATES = [
    os.environ.get("DATASET_PATH"),
    "backend/data/automation_data.csv",
    "data/automation_data.csv",
    "../data/automation_data.csv",
    "automation_data.csv",
]
csv_path = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("Não achei o dataset. Defina DATASET_PATH ou coloque 'automation_data.csv' em ./backend/data/.")

df = pd.read_csv(csv_path)

if "automatizar" not in df.columns:
    raise ValueError("Coluna 'automatizar' não encontrada.")

y = (
    df["automatizar"].astype(str).str.strip().str.lower()
      .map({"sim": 1, "nao": 0, "não": 0, "0": 0, "1": 1})
).astype(int).values

drop_cols = {"automatizar", "nome_tarefa", "id", "data", "data_registro"}
Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

cat_cols = list(Xdf.select_dtypes(include=["object", "category"]).columns)
num_cols = [c for c in Xdf.columns if c not in cat_cols]

pre = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

estimator = make_estimator = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

X_tr, X_te, y_tr, y_te = train_test_split(
    Xdf, y, test_size=0.30, random_state=42, stratify=y
)


pipe = Pipeline([("prep", pre), ("clf", estimator)])

cv_res = crossval_roc_calibrated(pipe, Xdf, y, k=5, random_state=42)
with open("metricas_cv.json", "w", encoding="utf-8") as f:
    json.dump(cv_res, f, ensure_ascii=False, indent=2)

mean_fpr = np.array(cv_res["roc_mean"]["fpr"])
mean_tpr = np.array(cv_res["roc_mean"]["tpr"])

plt.figure()
plt.plot(mean_fpr, mean_tpr,
         label=f"ROC média (AUC={cv_res['auc_mean']:.3f}±{cv_res['auc_std']:.3f})")
plt.fill_between(mean_fpr,
                 np.maximum(0, mean_tpr - 0.03), 
                 np.minimum(1, mean_tpr + 0.03),
                 alpha=0.15, label="Banda (±dp aprox.)")
plt.plot([0, 1], [0, 1], "--", label="Aleatório")
plt.xlabel("FPR (1-Especificidade)"); plt.ylabel("TPR (Sensibilidade)")
ci = cv_res["auc_ci95"]
plt.title(f"Curva ROC — Média 5-Fold\nAUC {cv_res['auc_mean']:.3f} "
          f"[{ci[0]:.3f}; {ci[1]:.3f}]")
plt.legend(); plt.grid(True, alpha=0.3)

thr = np.array(cv_res["oof"]["thresholds"])
fpr_oof = np.array(cv_res["oof"]["fpr"])
tpr_oof = np.array(cv_res["oof"]["tpr"])
tau = cv_res["youden_oof"]["tau"]
i = int(np.argmin(np.abs(thr - tau))) 
plt.scatter(fpr_oof[i], tpr_oof[i], s=60, marker="o", edgecolor="k", zorder=5,
            label=f"Youden OOF (τ*={tau:.3f})")
plt.legend()
plt.savefig("roc_cv_media.png", dpi=200, bbox_inches="tight")

hold = roc_holdout_calibrated(pipe, X_tr, y_tr, X_te, y_te)
with open("roc_holdout.json", "w", encoding="utf-8") as f:
    json.dump(hold, f, ensure_ascii=False, indent=2)

plt.figure()
plt.plot(hold["fpr"], hold["tpr"], label=f"Holdout ROC (AUC={hold['auc']:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Aleatório")

thr_h = np.array(hold["thresholds"])
fpr_h = np.array(hold["fpr"])
tpr_h = np.array(hold["tpr"])
tau_h = hold["tau_youden_holdout"]

j = np.argmin(np.abs(thr_h - tau_h))
plt.scatter(fpr_h[j], tpr_h[j], s=60, marker="o", edgecolor="k", zorder=5,
            label=f"Youden holdout (τ={tau_h:.3f}, sens={hold['sens_at_tau']:.2f}, esp={hold['spec_at_tau']:.2f})")
plt.legend()

plt.xlabel("FPR (1-Especificidade)"); plt.ylabel("TPR (Sensibilidade)")
plt.title("Curva ROC — Holdout (calibrada)")
plt.legend(); plt.grid(True, alpha=0.3)

thr_h = np.array(hold["thresholds"])
fpr_h = np.array(hold["fpr"])
tpr_h = np.array(hold["tpr"])
tau_h = hold["tau_youden_holdout"]
j = np.argmin(np.abs(thr_h - tau_h))
plt.scatter(fpr_h[j], tpr_h[j], s=60, marker="o", edgecolor="k", zorder=5,
            label=f"Youden holdout (τ={tau_h:.3f}, sens={hold['sens_at_tau']:.2f}, esp={hold['spec_at_tau']:.2f})")


tau_oof = cv_res["youden_oof"]["tau"]

y_prob = np.array(hold.get("_y_prob", []))  

plt.savefig("roc_holdout.png", dpi=200, bbox_inches="tight")

print("✅ Gerados: metricas_cv.json, roc_holdout.json, roc_cv_media.png, roc_holdout.png")



def plot_roc_folds(pipe, Xdf, y, k=5, seed=42, out_path="roc_cv_folds.png"):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    mean_fpr = np.linspace(0,1,101)
    tprs = []
    aucs = []

    plt.figure()
    for i, (tr, te) in enumerate(skf.split(Xdf, y), 1):
        model = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
        model.fit(Xdf.iloc[tr], y[tr])
        prob = model.predict_proba(Xdf.iloc[te])[:,1]

        
        fpr, tpr, _ = roc_curve(y[te], prob)
        auc = roc_auc_score(y[te], prob)
        aucs.append(auc)
        tpr_i = np.interp(mean_fpr, fpr, tpr); tpr_i[0]=0
        tprs.append(tpr_i)
        plt.plot(fpr, tpr, alpha=0.25)

    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Média (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
    plt.plot([0,1],[0,1],"--",label="Aleatório")
    plt.xlabel("FPR (1-Especificidade)"); plt.ylabel("TPR (Sensibilidade)")
    plt.title("Curvas ROC por fold (CV calibrada)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

plot_roc_folds(pipe, Xdf, y, k=5, seed=42)




def plot_calibration(pipe, Xdf, y, out_path="calibration_curve.png"):
    """Plota curva de calibração com Brier Score usando OOF manual."""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_oof = np.zeros_like(y, dtype=int)
    y_score_oof = np.zeros_like(y, dtype=float)

    for tr, te in skf.split(Xdf, y):
      
        model = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        model.fit(Xdf.iloc[tr], y[tr])
        prob = model.predict_proba(Xdf.iloc[te])[:, 1]

        y_true_oof[te] = y[te]
        y_score_oof[te] = prob

  
    brier = brier_score_loss(y_true_oof, y_score_oof)


    prob_true, prob_pred = calibration_curve(y_true_oof, y_score_oof, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label=f"Modelo (Brier={brier:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Perfeita")
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Frequência observada")
    plt.title("Curva de calibração (OOF, isotônica)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")



def plot_score_hist(pipe, Xdf, y, out_path="score_hist.png"):
    """Plota histogramas de scores OOF para classes positivas e negativas."""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_oof = np.zeros_like(y, dtype=int)
    y_score_oof = np.zeros_like(y, dtype=float)

    for tr, te in skf.split(Xdf, y):
        model = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        model.fit(Xdf.iloc[tr], y[tr])
        prob = model.predict_proba(Xdf.iloc[te])[:, 1]

        y_true_oof[te] = y[te]
        y_score_oof[te] = prob

    plt.figure()
    plt.hist(y_score_oof[y_true_oof == 1], bins=15, alpha=0.6, density=True, label="Positivos (y=1)")
    plt.hist(y_score_oof[y_true_oof == 0], bins=15, alpha=0.6, density=True, label="Negativos (y=0)")
    plt.xlabel("Score (probabilidade prevista)")
    plt.ylabel("Densidade")
    plt.title("Distribuição de scores por classe (OOF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")



def plot_pr(pipe, Xdf, y, out_path="pr_curve.png"):
    """Plota curva Precision–Recall com probabilidades OOF calibradas."""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_oof = np.zeros_like(y, dtype=int)
    y_score_oof = np.zeros_like(y, dtype=float)

    for tr, te in skf.split(Xdf, y):
        model = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        model.fit(Xdf.iloc[tr], y[tr])
        prob = model.predict_proba(Xdf.iloc[te])[:, 1]

        y_true_oof[te] = y[te]
        y_score_oof[te] = prob

    precision, recall, _ = precision_recall_curve(y_true_oof, y_score_oof)
    ap = average_precision_score(y_true_oof, y_score_oof)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision–Recall (OOF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")


total = len(y)
n1 = int((y == 1).sum())
n0 = int((y == 0).sum())
p1 = n1 / total if total else 0.0
p0 = n0 / total if total else 0.0

dist = {"total": total, "positivos_1_sim": n1, "negativos_0_nao": n0,
        "proporcao_1": round(p1, 4), "proporcao_0": round(p0, 4)}
Path("class_distribution.json").write_text(__import__("json").dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8")

plt.figure()
plt.bar(["1 (sim)", "0 (não)"], [n1, n0])
plt.title("Distribuição da variável-alvo (n = %d)" % total)
plt.ylabel("Contagem")
plt.grid(axis="y", alpha=0.3)
plt.savefig("class_balance.png", dpi=200, bbox_inches="tight")

print(f"▶ Distribuição da classe: 1(sim)={n1} ({p1:.1%}), 0(não)={n0} ({p0:.1%})")

plot_calibration(pipe, Xdf, y)
plot_score_hist(pipe, Xdf, y)
