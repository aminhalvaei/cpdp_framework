# cpdp/pipeline/runner.py
from typing import Any, Dict

from cpdp.pipeline.factories import (
    DatasetFactory,
    PreprocessorFactory,
    InstanceSelectorFactory,
    ProjectSelectorFactory,
    TransferFactory,
    ModelFactory,
    EvaluatorFactory,
)
from cpdp.utils.seed import set_seed


class ExperimentRunner:
    """
    High-level pipeline:
    1) Load source & target datasets
    2) Preprocess (fit on source, apply to both)
    3) Project selection
    4) Instance selection
    5) Wrap model with transfer method (if any)
    6) Train
    7) Evaluate on target
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        seed = config.get("seed", 42)
        set_seed(seed)

        self.source_dataset = DatasetFactory.create(config["source_dataset"])
        self.target_dataset = DatasetFactory.create(config["target_dataset"])

        self.preprocessor = PreprocessorFactory.create(config.get("preprocessing"))
        self.instance_selector = InstanceSelectorFactory.create(
            config.get("instance_selection")
        )
        self.project_selector = ProjectSelectorFactory.create(
            config.get("project_selection")
        )
        self.transfer = TransferFactory.create(config.get("transfer_learning"))
        self.model = ModelFactory.create(config["model"])
        self.evaluator = EvaluatorFactory.create(config["evaluation"])

    def run(self) -> Dict[str, float]:
        # 1. Load
        src = self.source_dataset.load()
        tgt = self.target_dataset.load()

        # 2. Preprocess
        src, tgt = self.preprocessor.fit_transform(src, tgt)

        # 3. Project selection
        if self.project_selector is not None:
            src = self.project_selector.select(src, tgt)

        # 4. Instance selection
        if self.instance_selector is not None:
            src = self.instance_selector.select(src, tgt)

        # 5. Transfer
        model = self.model
        if self.transfer is not None:
            model = self.transfer.wrap_model(model)

        # 6. Train
        # Simple baseline: standard supervised training on source
        model.train(src.features, src.labels)

        # 7. Evaluate on target
        metrics = self.evaluator.evaluate(model, tgt.features, tgt.labels)
        return metrics
