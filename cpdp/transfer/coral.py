# cpdp/transfer/coral.py
from typing import Optional

import numpy as np

from cpdp.models.base import BaseModel
from cpdp.pipeline.registry import register_transfer
from cpdp.transfer.base import BaseTransferMethod


def coral_transform(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """
    Very simple CORAL-like feature alignment (placeholder).
    This is not a full implementation but enough for experimentation.
    """
    # Compute covariance
    cov_src = np.cov(X_src, rowvar=False) + np.eye(X_src.shape[1])
    cov_tgt = np.cov(X_tgt, rowvar=False) + np.eye(X_tgt.shape[1])

    # Whitening transform for source
    U_s, S_s, _ = np.linalg.svd(cov_src)
    U_t, S_t, _ = np.linalg.svd(cov_tgt)

    sqrt_inv_cov_src = U_s @ np.diag(1.0 / np.sqrt(S_s)) @ U_s.T
    sqrt_cov_tgt = U_t @ np.diag(np.sqrt(S_t)) @ U_t.T

    A = sqrt_inv_cov_src @ sqrt_cov_tgt
    return (X_src - X_src.mean(axis=0)) @ A


class CORALWrappedModel(BaseModel):
    def __init__(self, base_model: BaseModel):
        self.base_model = base_model
        self._X_tgt: Optional[np.ndarray] = None

    def train(self, X_src, y_src) -> None:
        if self._X_tgt is None:
            # no target available; fallback to normal training
            self.base_model.train(X_src, y_src)
            return
        X_src_adapted = coral_transform(X_src, self._X_tgt)
        self.base_model.train(X_src_adapted, y_src)

    def set_target_features(self, X_tgt):
        self._X_tgt = X_tgt

    def predict(self, X_tgt):
        return self.base_model.predict(X_tgt)

    def predict_proba(self, X_tgt):
        return self.base_model.predict_proba(X_tgt)


@register_transfer("CORAL")
class CORALTransfer(BaseTransferMethod):
    def wrap_model(self, base_model: BaseModel) -> BaseModel:
        # Note: To use CORAL properly, you'll need to call
        # model.set_target_features(X_tgt) before training.
        # For baseline, you can ignore or integrate this later in runner.
        return CORALWrappedModel(base_model)
