# cpdp/models/sklearn_wrappers.py
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from cpdp.models.base import BaseModel
from cpdp.pipeline.registry import register_model


class SklearnModel(BaseModel):
    def __init__(self, estimator: Any):
        self.estimator = estimator

    def train(self, X_src, y_src) -> None:
        self.estimator.fit(X_src, y_src)

    def predict(self, X_tgt) -> np.ndarray:
        return self.estimator.predict(X_tgt)

    def predict_proba(self, X_tgt) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X_tgt)[:, 1]
        return super().predict_proba(X_tgt)


@register_model("RandomForest")
class RandomForestModel(SklearnModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs,
    ):
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        super().__init__(estimator)
