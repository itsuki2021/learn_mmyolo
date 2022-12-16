from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import Registry

DATASETS = Registry('dataset', parent=MMENGINE_DATASETS)
