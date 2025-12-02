# cpdp/selection/instance/none.py
from cpdp.data.base import DatasetSplit
from cpdp.pipeline.registry import register_instance_selector
from cpdp.selection.base import BaseInstanceSelector


@register_instance_selector("NoneInstanceSelection")
class NoneInstanceSelection(BaseInstanceSelector):
    def select(self, src: DatasetSplit, tgt: DatasetSplit) -> DatasetSplit:
        return src
