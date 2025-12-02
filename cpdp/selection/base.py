# cpdp/selection/base.py
from abc import ABC, abstractmethod

from cpdp.data.base import DatasetSplit


class BaseInstanceSelector(ABC):
    @abstractmethod
    def select(self, src: DatasetSplit, tgt: DatasetSplit) -> DatasetSplit:
        ...


class BaseProjectSelector(ABC):
    @abstractmethod
    def select(self, src: DatasetSplit, tgt: DatasetSplit) -> DatasetSplit:
        ...
