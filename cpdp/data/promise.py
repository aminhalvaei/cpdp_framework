# cpdp/data/promise.py
from cpdp.data.base import CsvDataset
from cpdp.pipeline.registry import register_dataset


@register_dataset("PromiseCsvDataset")
class PromiseCsvDataset(CsvDataset):
    """
    For PROMISE-like datasets stored as CSV.
    Currently identical to CsvDataset; you can extend this later
    to handle specific PROMISE conventions.
    """
    pass
