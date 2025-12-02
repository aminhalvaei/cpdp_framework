# cpdp/preprocessing/imbalance.py
from typing import Optional

from cpdp.data.base import DatasetSplit
from cpdp.pipeline.registry import register_preprocessor
from cpdp.preprocessing.base import BasePreprocessingStep


@register_preprocessor("NoImbalanceHandling")
class NoImbalanceHandlingStep(BasePreprocessingStep):
    """
    Baseline: do nothing.
    Later you can add SMOTE, undersampling, etc.
    """

    def fit(self, src: DatasetSplit, tgt: Optional[DatasetSplit] = None) -> None:
        return

    def transform(self, data: DatasetSplit) -> DatasetSplit:
        return data
