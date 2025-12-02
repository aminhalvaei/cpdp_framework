# cpdp/data/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cpdp.pipeline.registry import register_dataset


@dataclass
class DatasetSplit:
    features: np.ndarray
    labels: np.ndarray
    meta: Optional[Dict[str, Any]] = None


class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> DatasetSplit:
        """Load dataset and return as DatasetSplit."""
        raise NotImplementedError


@register_dataset("CsvDataset")
class CsvDataset(BaseDataset):
    """
    Simple baseline: load from a CSV file.

    Assumes:
    - One column is the label column (binary 0/1)
    - All other numeric columns are features.
    """

    def __init__(
        self,
        path: str,
        label_column: str,
        project_name: Optional[str] = None,
    ):
        self.path = path
        self.label_column = label_column
        self.project_name = project_name or path

    def load(self) -> DatasetSplit:
        df = pd.read_csv(self.path)
        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in {self.path}"
            )

        labels = df[self.label_column].values.astype(int)
        features = df.drop(columns=[self.label_column]).values.astype(float)
        meta = {"project": self.project_name, "path": self.path, "columns": df.columns}
        return DatasetSplit(features=features, labels=labels, meta=meta)
