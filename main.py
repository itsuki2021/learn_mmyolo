import torch
import mmcv
import mmengine
import mmdet
import mmyolo

if __name__ == '__main__':
    print(f"torch.__version__:{torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"mmcv.__version__: {mmcv.__version__}")
    print(f"mmengine.__version__: {mmengine.__version__}")
    print(f"mmdet.__version__: {mmdet.__version__}")
    print(f"mmyolo.__version__: {mmyolo.__version__}")
    print("Done.")
