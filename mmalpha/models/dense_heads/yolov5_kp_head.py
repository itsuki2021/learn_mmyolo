import math
from typing import Union, Sequence, Tuple, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmengine.dist import get_dist_info
from mmdet.utils import (ConfigType, OptInstanceList,
                         OptMultiConfig)
from mmdet.models.utils import multi_apply
from mmyolo.models.dense_heads import YOLOv5Head
from mmyolo.models.utils import make_divisible
from mmalpha.registry import MODELS


@MODELS.register_module()
class YOLOv5KeyPointHeadModule(BaseModule):
    """YOLOv5KeypointHeadModule head module used in `YOLOv5`.
        from mmyolo.models.dense_heads.YOLOv5HeadModule

    Args:
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        num_keypoint (int) : The number of keypoint at a point on the feature grid.
        num_keypoint_visible (int) : The number of keypoint visible level.
            Defaults to 3 (Invisible, Occlusion, Visible)
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        num_base_priors:int: The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to (8, 16, 32).
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[int, Sequence],
                 num_keypoint: int,
                 num_keypoint_visible: int = 3,
                 widen_factor: float = 1.0,
                 num_base_priors: int = 3,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = 1  # keypoint detector requires single class
        self.num_keypoint = num_keypoint
        self.num_keypoint_visible = num_keypoint_visible
        self.widen_factor = widen_factor

        self.featmap_strides = featmap_strides
        self.num_out_attrib = 5 + self.num_classes + self.num_keypoint * (self.num_keypoint_visible + 2)
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors

        if isinstance(in_channels, int):
            self.in_channels = [make_divisible(in_channels, widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [
                make_divisible(i, widen_factor) for i in in_channels
            ]

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_base_priors * self.num_out_attrib,
                                  (1, 1))

            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super().init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999))

            mi.bias.data = b.view(-1)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.convs_pred)

    def forward_single(self, x: Tensor, convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib, ny, nx)  # B, P, C, H, W

        # pred_map: [tx, ty, tw, th, objectness, cls, [v11, v12, v13, v21, v22, v23, ...,tx1, ty1, tx2, ty2, ...]]
        len_kp_visible = self.num_keypoint * self.num_keypoint_visible
        bbox_preds = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        objectnesses = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)
        cls_scores = pred_map[:, :, 5:5 + self.num_classes, ...].reshape(bs, -1, ny, nx)
        kps_visible_preds = pred_map[:, :, 5 + self.num_classes:5 + self.num_classes + len_kp_visible, ...].reshape(bs,
                                                                                                                    -1,
                                                                                                                    ny,
                                                                                                                    nx)
        kps_reg_preds = pred_map[:, :, 5 + self.num_classes + len_kp_visible:, ...].reshape(bs, -1, ny, nx)

        return cls_scores, bbox_preds, objectnesses, kps_visible_preds, kps_reg_preds


@MODELS.register_module()
class YOLOv5KeyPointHead(YOLOv5Head):
    """YOLOv5KeyPointHead head used in `YOLOv5`.

    Args:
        loss_kps_visible (:obj:`ConfigDict` or dict): Config of keypoint visible loss.
        loss_kps_reg (:obj:`ConfigDict` or dict): Config of keypoint regression loss.
    """

    def __init__(self,
                 loss_kps_visible: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 loss_kps_reg: ConfigType = dict(
                     type='mmdet.MSELoss'),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_keypoint_visible = self.head_module.num_keypoint_visible
        self.num_keypoint = self.head_module.num_keypoint

        self.loss_kps_visible: nn.Module = MODELS.build(loss_kps_visible)
        self.loss_kps_reg: nn.Module = MODELS.build(loss_kps_reg)

    def loss(self, x: Tuple[Tensor], batch_data_samples: dict) -> dict:
        """Perform forward propagation and loss calculations of the keypoint detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.
            batch_data_samples (dict): The Data Samples from
                preprocessor (YOLOv5KeyPointDataPreprocessor) output.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores, bbox_preds, objectnesses, kps_visible_preds, kps_reg_preds = self(x)
        batch_gt_instances = batch_data_samples['bboxes_labels']
        batch_img_metas = batch_data_samples['img_metas']
        losses = self.loss_by_feat(cls_scores=cls_scores,
                                   bbox_preds=bbox_preds,
                                   objectnesses=objectnesses,
                                   kps_visible_preds=kps_visible_preds,
                                   kps_reg_preds=kps_reg_preds,
                                   batch_gt_instances=batch_gt_instances,
                                   batch_img_metas=batch_img_metas)

        return losses

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None,
            kps_visible_preds: Optional[List[Tensor]] = None,
            kps_reg_preds: Optional[List[Tensor]] = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            kps_visible_preds (Sequence[Tensor]): keypoint visible for each scale level,
                each is a 4D-tensor, the channel number is num_priors * num_keypoints * num_keypoints_visible.
            kps_reg_preds (Sequence[Tensor]): keypoint coordinate regression for each
                scale leve, each is a 4D-tensor, the channel number is num_priors * num_keypoints * 2
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(batch_gt_instances, batch_img_metas)

        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_kps_visible = torch.zeros(1, device=device)
        loss_kps_reg = torch.zeros(1, device=device)
        scaled_factor = torch.ones(batch_targets_normed.shape[-1], device=device)

        batch_size = bbox_preds[0].shape[0]
        for i in range(self.num_levels):
            _, _, h, w = bbox_preds[i].shape
            target_obj = torch.zeros_like(objectnesses[i])

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_kps_visible += kps_visible_preds[i].sum() * 0
                loss_kps_reg += kps_reg_preds[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            # feature map scale whwh
            scaled_factor[2:6:2] = torch.tensor(bbox_preds[i].shape[3])  # bbox x and w
            scaled_factor[3:6:2] = torch.tensor(bbox_preds[i].shape[2])  # bbox y and h
            scaled_factor[6:-1:3] = torch.tensor(bbox_preds[i].shape[3])  # keypoint x
            scaled_factor[7:-1:3] = torch.tensor(bbox_preds[i].shape[2])  # keypoint y
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_bboxes, batch_id + label + xywh + xyv * num_keypoint + prior_id)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            wh_ratio = batch_targets_scaled[..., 4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_kps_visible += kps_visible_preds[i].sum() * 0
                loss_kps_reg += kps_reg_preds[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                continue

            # 3. Positive samples with additional neighbors

            # check the left, up, right, bottom sides of the
            # targets grid, and determine whether assigned
            # them as positive samples as well.
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) & (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) & (grid_xy > 1)).T
            # neighbors girds: center (always select), left, up, right, bottom
            offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat((5, 1, 1))[offset_inds]
            # offsets of neighbor grids to center grid
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1], 1)[offset_inds]

            # prepare pred results and positive sample indexes to
            # calculate class loss, bbox loss and keypoint loss
            img_class_inds = batch_targets_scaled[:, :2]
            grid_xy = batch_targets_scaled[:, 2:4]
            grid_wh = batch_targets_scaled[:, 4:6]
            grid_kps_x = batch_targets_scaled[:, 6:-1:3]
            grid_kps_y = batch_targets_scaled[:, 7:-1:3]
            grid_kps_xy = torch.cat((grid_kps_x, grid_kps_y), 1)
            priors_inds = batch_targets_scaled[:, -1, None]
            priors_inds, (img_inds, class_inds) = priors_inds.long().view(-1), img_class_inds.long().T

            # center to neighbors
            grid_xy_long = (grid_xy - retained_offsets * self.near_neighbor_thr).long()
            grid_kps_xy_long = (
                    grid_kps_xy - retained_offsets.repeat(1, self.num_keypoint) * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh), 1)

            # 4. Calculate loss
            # bbox loss
            retained_bbox_pred = bbox_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            priors_base_sizes_i = priors_base_sizes_i[priors_inds]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i)
            loss_box_i, iou = self.loss_bbox(decoded_bbox_pred, bboxes_targets)
            loss_box += loss_box_i

            # obj loss
            iou = iou.detach().clamp(0)
            target_obj[img_inds, priors_inds, grid_y_inds,
                       grid_x_inds] = iou.type(target_obj.dtype)
            loss_obj += self.loss_obj(objectnesses[i],
                                      target_obj) * self.obj_level_weights[i]

            # cls loss
            if self.num_classes > 1:
                pred_cls_scores = cls_scores[i].reshape(
                    batch_size, self.num_base_priors, -1, h,
                    w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_class = torch.full_like(pred_cls_scores, 0.)
                target_class[range(batch_targets_scaled.shape[0]),
                             class_inds] = 1.
                loss_cls += self.loss_cls(pred_cls_scores, target_class)
            else:
                loss_cls += cls_scores[i].sum() * 0

            # keypoint visible loss
            if self.num_keypoint_visible > 1:
                retained_kps_visible_preds = kps_visible_preds[i].reshape(
                    batch_size, self.num_base_priors, -1, h,
                    w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_kp_visible = torch.full_like(retained_kps_visible_preds, 0.)
                instance_inds = torch.arange(0, batch_targets_scaled.shape[0]).repeat(self.num_keypoint, 1).long().T
                kps_visible_inds = batch_targets_scaled[:, 8:-1:3] + \
                                   torch.arange(0, self.num_keypoint, device=device) * self.num_keypoint_visible
                kps_visible_inds = kps_visible_inds.long()
                target_kp_visible[instance_inds, kps_visible_inds] = 1.
                loss_kps_visible += self.loss_kps_visible(retained_kps_visible_preds, target_kp_visible)
            else:
                loss_kps_visible += kps_visible_preds[i].sum() * 0

            # keypoint regression loss
            kps_reg_targets = (grid_kps_xy - grid_kps_xy_long).float()
            retained_kps_reg_preds = kps_reg_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            loss_kps_reg += self.loss_kps_reg(retained_kps_reg_preds, kps_reg_targets)

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size,
            loss_kps_visible=loss_kps_visible * batch_size * world_size,
            loss_kps_reg=loss_kps_reg * batch_size * world_size
        )

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        kps_visible: Optional[List[Tensor]] = None,
                        kps_reg: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        # todo
        return []

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_bboxes_xyxy = batch_gt_instances[:, 2:6]
            xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
            gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
            gt_bboxes_xywh[:, 1::2] /= img_shape[0]
            gt_bboxes_xywh[:, 0::2] /= img_shape[1]
            batch_gt_instances[:, 2:6] = gt_bboxes_xywh

            batch_gt_instances[:, 7::3] /= img_shape[0]  # keypoint y
            batch_gt_instances[:, 6::3] /= img_shape[1]  # keypoint x

            # (num_base_priors, num_bboxes, batch_id + label_id + xywh + xyv * num_keypoints)
            batch_targets_normed = batch_gt_instances.repeat(
                self.num_base_priors, 1, 1)
        else:
            batch_target_list = []
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                bboxes = gt_instances.bboxes
                keypoints = gt_instances.keypoints  # format=xyv
                labels = gt_instances.labels

                xy1, xy2 = bboxes.split((2, 2), dim=-1)
                bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                # normalized to 0-1
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]
                keypoints[:, 1::3] /= img_shape[0]
                keypoints[:, 0::3] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, normed_bbox, normed_keypoint)
                target = torch.cat((index, labels[:, None].float(), bboxes, keypoints),
                                   dim=1)
                batch_target_list.append(target)

            # (num_base_priors, num_bboxes, batch_id + label_id + xywh + xyv * num_keypoints)
            batch_targets_normed = torch.cat(
                batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)

        # (num_base_priors, num_bboxes, 1)
        batch_targets_prior_inds = self.prior_inds.repeat(
            1, batch_targets_normed.shape[1])[..., None]
        # (num_base_priors, num_bboxes, N)
        # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h,
        #  keypoint_x, keypoint_y, keypoint_visible) * num_keypoint, prior_ind)
        batch_targets_normed = torch.cat(
            (batch_targets_normed, batch_targets_prior_inds), 2)
        return batch_targets_normed
