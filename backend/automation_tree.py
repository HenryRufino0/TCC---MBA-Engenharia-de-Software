from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

RANDOM_SEED = 42  


BASE_COLS = [
    "frequencia",
    "tempo",
    "complexidade",  
    "importancia",   # Essa aqui desconsiderei
    "urgencia",      
    "ferramentas",
    "volume",
    "colaboradores",
    "automatizar",   
]

TARGET = "automatizar"  


class AutomationTree:
    def __init__(
        self,
        *,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_impurity_decrease: float = 0.001,
        ccp_alpha: Optional[float] = None,
        random_state: int = RANDOM_SEED,
    ):
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha_cfg = ccp_alpha

        self.model: Optional[DecisionTreeClassifier] = None

        
        self._onehot_cols: List[str] = []
        self.feature_order_: List[str] = []

        self.is_trained: bool = False
        self.metrics_: Optional[Dict[str, Any]] = None

    #Validação dos dados

    @staticmethod
    def _norm_text(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip().str.lower()
        s = s.replace({"média": "media", "não": "nao"})
        return s

    def _validate_cols(self, df: pd.DataFrame):
        missing = set(BASE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Faltam colunas: {', '.join(sorted(missing))}")

    
    def _prepare_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # normalizando os textos
        for c in ["complexidade", "urgencia", "importancia", TARGET]:
            df[c] = self._norm_text(df[c])

        # One-hot (nao tem mais importancia)
        df_ohe = pd.get_dummies(
            df,
            columns=["complexidade", "urgencia"],
            prefix=["complexidade", "urgencia"],
            drop_first=False,
        )

        self._onehot_cols = sorted(
            [c for c in df_ohe.columns if c.startswith("complexidade_") or c.startswith("urgencia_")]
        )

        base_numeric = ["frequencia", "tempo", "ferramentas", "volume", "colaboradores"]
        self.feature_order_ = base_numeric + self._onehot_cols

        df_ohe[TARGET] = df_ohe[TARGET].map({"sim": 1, "nao": 0})
        if df_ohe[TARGET].isna().any():
            raise ValueError("Valores de 'automatizar' devem ser 'sim' ou 'nao'.")

        return df_ohe

    
    def _prepare_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ["complexidade", "urgencia", "importancia"]:
            df[c] = self._norm_text(df[c])

        df_ohe = pd.get_dummies(
            df,
            columns=["complexidade", "urgencia"],
            prefix=["complexidade", "urgencia"],
            drop_first=False,
        )

        
        for col in self._onehot_cols:
            if col not in df_ohe.columns:
                df_ohe[col] = 0

        base_numeric = ["frequencia", "tempo", "ferramentas", "volume", "colaboradores"]
        df_ohe = df_ohe.reindex(columns=base_numeric + self._onehot_cols, fill_value=0)
        return df_ohe.astype(float)



    # PODA dos Dados



    def _pick_ccp_alpha(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> float:
        if self.ccp_alpha_cfg is not None:
            return float(self.ccp_alpha_cfg)
        tmp = DecisionTreeClassifier(random_state=self.random_state)
        path = tmp.cost_complexity_pruning_path(X_tr, y_tr)
        alphas = path.ccp_alphas
        if alphas.size <= 1:
            return 0.0
        return float(np.quantile(alphas[:-1], 0.5))  



    # Treinamento e metrias



    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        np.random.seed(self.random_state)
        self._validate_cols(df)

        df_enc = self._prepare_fit(df)

        X_all = df_enc[self.feature_order_].astype(float)
        y_all = df_enc[TARGET].astype(int)

        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all,
            y_all,
            test_size=0.30,
            stratify=y_all,
            random_state=self.random_state,
        )

        alpha = self._pick_ccp_alpha(X_tr, y_tr)

        self.model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=alpha,
            class_weight="balanced",   # esse balanceia o alvo se estiver desbalanceado
            max_features="sqrt",       # aqui força considerar subconjuntos de features
            random_state=self.random_state,
        )
        self.model.fit(X_tr, y_tr)
        self.is_trained = True

        # Teste de métricas 
        y_pred = self.model.predict(X_te)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        acc = accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="binary", pos_label=1, zero_division=0
        )

        imps = {feat: float(w) for feat, w in zip(self.feature_order_, self.model.feature_importances_)}

        self.metrics_ = {
            "random_state": self.random_state,
            "params": {
                "criterion": "gini",
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "min_impurity_decrease": self.min_impurity_decrease,
                "ccp_alpha": float(alpha),
                "class_weight": "balanced",
                "max_features": "sqrt",
                "features_usadas": self.feature_order_,
                "feature_importances": imps,
                "observacao": "IMPORTANTE: 'importancia' NÃO é usada como feature para evitar dominância.",
            },
            "test_metrics": {
                "accuracy": float(acc),
                "precision_sim": float(prec),
                "recall_sim": float(rec),
                "f1_sim": float(f1),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
        }
        return self.metrics_



    # Predict


    def _prepare_single_input(
        self,
        *,
        frequencia: int,
        tempo: float,
        complexidade: str,
        importancia: str,  # IGNORADO (nao usa mais)
        urgencia: str,
        ferramentas: int,
        volume: int,
        colaboradores: int,
    ) -> pd.DataFrame:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Modelo ainda não foi treinado.")

        df = pd.DataFrame([{
            "frequencia": frequencia,
            "tempo": tempo,
            "complexidade": complexidade,
            "importancia": importancia,  
            "urgencia": urgencia,
            "ferramentas": int(ferramentas),
            "volume": int(volume),
            "colaboradores": int(colaboradores),
        }])
        X = self._prepare_transform(df)
        X = X.reindex(columns=self.feature_order_, fill_value=0).astype(float)
        return X

    def predict(
        self,
        frequencia: int,
        tempo: float,
        complexidade: str,
        importancia: str,
        urgencia: str,
        ferramentas: int,
        volume: int,
        colaboradores: int,
    ) -> str:
        X = self._prepare_single_input(
            frequencia=frequencia,
            tempo=tempo,
            complexidade=complexidade,
            importancia=importancia,
            urgencia=urgencia,
            ferramentas=ferramentas,
            volume=volume,
            colaboradores=colaboradores,
        )
        pred = self.model.predict(X)[0]
        return "sim" if int(pred) == 1 else "nao"

    def predict_proba(
        self,
        frequencia: int,
        tempo: float,
        complexidade: str,
        importancia: str,
        urgencia: str,
        ferramentas: int,
        volume: int,
        colaboradores: int,
    ) -> float:
        X = self._prepare_single_input(
            frequencia=frequencia,
            tempo=tempo,
            complexidade=complexidade,
            importancia=importancia,
            urgencia=urgencia,
            ferramentas=ferramentas,
            volume=volume,
            colaboradores=colaboradores,
        )
        proba_sim = self.model.predict_proba(X)[0][1]
        return float(proba_sim)

    

    #salvar as metricas
    
    def metrics(self) -> Optional[Dict[str, Any]]:
        return self.metrics_
