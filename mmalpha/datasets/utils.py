from typing import Sequence

import torch

from mmengine.dataset import COLLATE_FUNCTIONS


@COLLATE_FUNCTIONS.register_module()
def yolov5_kp_collate(data_batch: Sequence) -> dict:
    """Rewrite collate_fn to get faster training speed."""
    batch_imgs = []
    batch_bboxes_kps_labels = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']

        # use 'Tensor' instead of 'HorizontalBoxes' in mmyolo.datasets.utils.yolov5_collate
        gt_bboxes = datasamples.gt_instances.bboxes

        gt_labels = datasamples.gt_instances.labels
        gt_keypoints = datasamples.gt_instances.keypoints
        flatten_gt_keypoints = gt_keypoints.reshape(gt_keypoints.shape[0], -1)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes, flatten_gt_keypoints),
                                  dim=1)
        batch_bboxes_kps_labels.append(bboxes_labels)

        batch_imgs.append(inputs)
    return {
        'inputs': torch.stack(batch_imgs, 0),
        'data_samples': torch.cat(batch_bboxes_kps_labels, 0)
    }