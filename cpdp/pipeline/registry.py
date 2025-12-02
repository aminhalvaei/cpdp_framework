from typing import Dict, Type

# Simple registries for plugin-like architecture
DATASET_REGISTRY: Dict[str, Type] = {}
PREPROCESSOR_REGISTRY: Dict[str, Type] = {}
INSTANCE_SELECTOR_REGISTRY: Dict[str, Type] = {}
PROJECT_SELECTOR_REGISTRY: Dict[str, Type] = {}
TRANSFER_REGISTRY: Dict[str, Type] = {}
MODEL_REGISTRY: Dict[str, Type] = {}
EVALUATOR_REGISTRY: Dict[str, Type] = {}


def _make_register(registry: Dict[str, Type]):
    def register(name: str):
        def decorator(cls):
            registry[name] = cls
            return cls
        return decorator
    return register


register_dataset = _make_register(DATASET_REGISTRY)
register_preprocessor = _make_register(PREPROCESSOR_REGISTRY)
register_instance_selector = _make_register(INSTANCE_SELECTOR_REGISTRY)
register_project_selector = _make_register(PROJECT_SELECTOR_REGISTRY)
register_transfer = _make_register(TRANSFER_REGISTRY)
register_model = _make_register(MODEL_REGISTRY)
register_evaluator = _make_register(EVALUATOR_REGISTRY)
