# cpdp/pipeline/factories.py
from typing import Any, Dict, List, Optional

from .registry import (
    DATASET_REGISTRY,
    PREPROCESSOR_REGISTRY,
    INSTANCE_SELECTOR_REGISTRY,
    PROJECT_SELECTOR_REGISTRY,
    TRANSFER_REGISTRY,
    MODEL_REGISTRY,
    EVALUATOR_REGISTRY,
)
from cpdp.preprocessing.base import SequentialPreprocessor


def _create_from_registry(registry: dict, config: Optional[Dict[str, Any]]) -> Any:
    if config is None:
        return None
    name = config.get("name")
    params = config.get("params", {}) or {}
    if name not in registry:
        raise ValueError(f"Unknown component '{name}' in registry.")
    cls = registry[name]
    return cls(**params)


class DatasetFactory:
    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        return _create_from_registry(DATASET_REGISTRY, config)


class PreprocessorFactory:
    @staticmethod
    def create(config: Optional[Dict[str, Any]]) -> SequentialPreprocessor:
        if not config:
            return SequentialPreprocessor([])
        steps_cfg: List[Dict[str, Any]] = config.get("steps", [])
        steps = []
        for step_cfg in steps_cfg:
            name = step_cfg["name"]
            params = step_cfg.get("params", {}) or {}
            if name not in PREPROCESSOR_REGISTRY:
                raise ValueError(f"Unknown preprocessing step '{name}'")
            cls = PREPROCESSOR_REGISTRY[name]
            steps.append(cls(**params))
        return SequentialPreprocessor(steps)


class InstanceSelectorFactory:
    @staticmethod
    def create(config: Optional[Dict[str, Any]]) -> Any:
        return _create_from_registry(INSTANCE_SELECTOR_REGISTRY, config)


class ProjectSelectorFactory:
    @staticmethod
    def create(config: Optional[Dict[str, Any]]) -> Any:
        return _create_from_registry(PROJECT_SELECTOR_REGISTRY, config)


class TransferFactory:
    @staticmethod
    def create(config: Optional[Dict[str, Any]]) -> Any:
        return _create_from_registry(TRANSFER_REGISTRY, config)


class ModelFactory:
    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        return _create_from_registry(MODEL_REGISTRY, config)


class EvaluatorFactory:
    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        return _create_from_registry(EVALUATOR_REGISTRY, config)
