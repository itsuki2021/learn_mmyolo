_base_ = [
    './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
]

load_from = 'checkpoints/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'

# Dataset and Dataloader
data_root = 'data/dmcode/'
metainfo = {
    'CLASSES': ('dmcode', ),
    'PALETTE': [
        (220, 20, 60),
    ]
}
train_batch_size_per_gpu = 4
train_num_workers = 2
dataset_type = 'mmalpha.YOLOv5CocoKeyPointDataset'
pipeline = [
    dict(type='mmengine.LoadImageFromFile'),
    dict(type='mmengine.LoadAnnotations', with_keypoints=True),
    dict(type='mmengine.Resize', scale=(640, 640)),
    dict(type='mmalpha.PackDetKpsInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/'),
        ann_file='annotations/coco/train.json',
        pipeline=pipeline),
    collate_fn=dict(type='yolov5_kp_collate')
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/'),
        ann_file='annotations/coco/train.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/coco/train.json')
test_evaluator = val_evaluator

# model
model = dict(
    type='mmalpha.YOLOKeyPointDetector',
    bbox_head=dict(
        type='mmalpha.YOLOv5KeyPointHead',
        head_module=dict(
            _delete_=True,
            type='mmalpha.YOLOv5KeyPointHeadModule',
            in_channels=[256, 512, 1024],
            widen_factor={{_base_.widen_factor}},
            num_keypoint=4,
            num_keypoint_visible=3),
        loss_kps_visible=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        loss_kps_reg=dict(type='mmdet.MSELoss')
    )
)

default_hooks = dict(logger=dict(interval=1))
