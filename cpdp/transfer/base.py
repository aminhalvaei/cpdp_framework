# cpdp/transfer/base.py
from abc import ABC, abstractmethod

from cpdp.models.base import BaseModel
from cpdp.pipeline.registry import register_transfer


class BaseTransferMethod(ABC):
    @abstractmethod
    def wrap_model(self, base_model: BaseModel) -> BaseModel:
        ...


@register_transfer("NoneTransfer")
class NoneTransfer(BaseTransferMethod):
    """
    Baseline: do not modify the model or data.
    Kept for config consistency.
    """

    def wrap_model(self, base_model: BaseModel) -> BaseModel:
        return base_model
