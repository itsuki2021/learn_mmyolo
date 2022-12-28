from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry


RUNNERS = Registry('runner', parent=MMENGINE_RUNNERS)
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS)
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS)
MODELS = Registry('model', parent=MMENGINE_MODELS)
