# cpdp/preprocessing/scaling.py
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from cpdp.data.base import DatasetSplit
from cpdp.pipeline.registry import register_preprocessor
from cpdp.preprocessing.base import BasePreprocessingStep


@register_preprocessor("StandardScaler")
class StandardScalerStep(BasePreprocessingStep):
    def __init__(self):
        self._scaler: Optional[StandardScaler] = None

    def fit(self, src: DatasetSplit, tgt: Optional[DatasetSplit] = None) -> None:
        self._scaler = StandardScaler()
        self._scaler.fit(src.features)

    def transform(self, data: DatasetSplit) -> DatasetSplit:
        if self._scaler is None:
            raise RuntimeError("StandardScalerStep must be fitted before transform.")
        features = self._scaler.transform(data.features)
        return DatasetSplit(features=features, labels=data.labels, meta=data.meta)
