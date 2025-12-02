# cpdp/evaluation/classification.py
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from cpdp.evaluation.base import BaseEvaluator
from cpdp.models.base import BaseModel
from cpdp.pipeline.registry import register_evaluator


@register_evaluator("BinaryClassificationEvaluator")
class BinaryClassificationEvaluator(BaseEvaluator):
    def __init__(self, metrics: List[str]):
        self.metrics = metrics

    def evaluate(self, model: BaseModel, X_tgt, y_tgt) -> Dict[str, float]:
        y_tgt = np.asarray(y_tgt)

        results: Dict[str, float] = {}
        y_pred = model.predict(X_tgt)

        if "auc" in self.metrics:
            try:
                y_proba = model.predict_proba(X_tgt)
                results["auc"] = float(roc_auc_score(y_tgt, y_proba))
            except Exception:
                # fallback: no probability-based AUC
                pass

        if "f1" in self.metrics:
            results["f1"] = float(f1_score(y_tgt, y_pred))

        if "precision" in self.metrics:
            results["precision"] = float(precision_score(y_tgt, y_pred))

        if "recall" in self.metrics:
            results["recall"] = float(recall_score(y_tgt, y_pred))

        return results
