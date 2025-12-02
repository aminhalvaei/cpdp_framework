# cpdp/preprocessing/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from cpdp.data.base import DatasetSplit


class BasePreprocessingStep(ABC):
    @abstractmethod
    def fit(self, src: DatasetSplit, tgt: Optional[DatasetSplit] = None) -> None:
        ...

    @abstractmethod
    def transform(self, data: DatasetSplit) -> DatasetSplit:
        ...


class SequentialPreprocessor:
    def __init__(self, steps: List[BasePreprocessingStep]):
        self.steps = steps

    def fit_transform(
        self, src: DatasetSplit, tgt: Optional[DatasetSplit] = None
    ) -> Tuple[DatasetSplit, Optional[DatasetSplit]]:
        for step in self.steps:
            step.fit(src, tgt)
            src = step.transform(src)
            if tgt is not None:
                tgt = step.transform(tgt)
        return src, tgt

    def transform(self, data: DatasetSplit) -> DatasetSplit:
        for step in self.steps:
            data = step.transform(data)
        return data
