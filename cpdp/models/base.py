# cpdp/models/base.py
from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_src, y_src) -> None:
        ...

    @abstractmethod
    def predict(self, X_tgt) -> Any:
        ...

    def predict_proba(self, X_tgt):
        # Optional; default is to return hard predictions as "probabilities"
        preds = self.predict(X_tgt)
        return preds
