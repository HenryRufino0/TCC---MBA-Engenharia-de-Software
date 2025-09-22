import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from metrics_cv import crossval_cart, roc_holdout


df = pd.read_csv("backend/data/automation_data.csv")


y = (
    df["automatizar"]
    .astype(str).str.strip().str.lower()
    .map({"sim": 1, "não": 0, "nao": 0, "0": 0, "1": 1})
).values


X = df.drop(columns=["automatizar"]).copy()


if "complexidade" in X.columns:
    X["complexidade"] = (
        X["complexidade"].astype(str).str.strip().str.lower()
        .map({"baixa": 1, "media": 2, "média": 2, "alta": 3})
        .fillna(2)
        .astype(int)
    )


for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].astype("category").cat.codes

X = X.values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)


model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_train, y_train)


cv_res = crossval_cart(model, X_train, y_train, k=5)
with open("metricas_cv.json", "w", encoding="utf-8") as f:
    json.dump(cv_res, f, ensure_ascii=False, indent=2)

roc_res = roc_holdout(model, X_test, y_test)
with open("roc_holdout.json", "w", encoding="utf-8") as f:
    json.dump(roc_res, f, ensure_ascii=False, indent=2)


fpr, tpr = cv_res["roc_mean"]["fpr"], cv_res["roc_mean"]["tpr"]
plt.figure()
plt.plot(fpr, tpr, label=f"ROC média (AUC={cv_res['avg']['auc']:.3f}±{cv_res['std']['auc']:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Aleatório")
plt.xlabel("FPR (1-Especificidade)"); plt.ylabel("TPR (Sensibilidade)")
plt.title("Curva ROC — Média 5-Fold")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("roc_cv_media.png", dpi=200, bbox_inches="tight")


fpr, tpr, auc = roc_res["fpr"], roc_res["tpr"], roc_res["auc"]
plt.figure()
plt.plot(fpr, tpr, label=f"Holdout ROC (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Aleatório")
plt.xlabel("FPR (1-Especificidade)"); plt.ylabel("TPR (Sensibilidade)")
plt.title("Curva ROC — Holdout")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("roc_holdout.png", dpi=200, bbox_inches="tight")

print("✅ Gerados: metricas_cv.json, roc_holdout.json, roc_cv_media.png, roc_holdout.png")
