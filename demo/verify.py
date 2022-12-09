from mmdet.apis import init_detector, inference_detector
from mmyolo.utils import register_all_modules

if __name__ == '__main__':
    register_all_modules()
    config_file = 'configs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
    checkpoint_file = 'checkpoints/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
    result = inference_detector(model, 'data/demo.jpg')
    print(result)
    print("Done.")
