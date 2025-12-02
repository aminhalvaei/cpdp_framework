# cpdp/selection/project/all_projects.py
from cpdp.data.base import DatasetSplit
from cpdp.pipeline.registry import register_project_selector
from cpdp.selection.base import BaseProjectSelector


@register_project_selector("AllProjectsSelector")
class AllProjectsSelector(BaseProjectSelector):
    def select(self, src: DatasetSplit, tgt: DatasetSplit) -> DatasetSplit:
        # Baseline: keep all source projects as-is
        return src
