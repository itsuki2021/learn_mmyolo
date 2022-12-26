from mmengine.registry import Registry
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import MODELS as MMENGINE_MODELS


DATASETS = Registry('dataset', parent=MMENGINE_DATASETS)
TRANSFORMS = Registry('transforms', parent=MMENGINE_TRANSFORMS)
MODELS = Registry('models', parent=MMENGINE_MODELS)
