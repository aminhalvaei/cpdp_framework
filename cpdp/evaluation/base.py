# cpdp/evaluation/base.py
from abc import ABC, abstractmethod
from typing import Dict

from cpdp.models.base import BaseModel


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model: BaseModel, X_tgt, y_tgt) -> Dict[str, float]:
        ...
