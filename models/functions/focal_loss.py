import jittor as jt
import jittor.nn as nn


class FocalLoss(nn.Module):
    #def __init__(self):

    def execute(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = jt.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                alpha_factor = jt.ones(classification.shape) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * jt.pow(focal_weight, gamma)
                bce = -(jt.log(1.0 - classification))
                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(jt.array(0).float())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max = jt.max(IoU , dim = 1)
            IoU_argmax = jt.argmax(IoU , dim = 1)
            
            # compute the loss for classification
            targets = jt.ones(classification.shape) * -1

            targets[IoU_max < 0.4, :] = 0

            positive_indices = IoU_max >= 0.5

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = jt.ones(targets.shape) * alpha

            alpha_factor = jt.where(targets == 1., alpha_factor, 1. - alpha_factor)
            focal_weight = jt.where(targets == 1., 1. - classification, classification)
            focal_weight = alpha_factor * jt.pow(focal_weight, gamma)

            bce = -(targets * jt.log(classification) + (1.0 - targets) * jt.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = jt.where(targets != -1.0, cls_loss, jt.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/jt.clamp(num_positive_anchors.float(), min_v=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = jt.clamp(gt_widths, min_v=1)
                gt_heights = jt.clamp(gt_heights, min_v=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = jt.log(gt_widths / anchor_widths_pi)
                targets_dh = jt.log(gt_heights / anchor_heights_pi)

                targets = jt.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = jt.transpose(targets)

                targets = targets / jt.array([[0.1, 0.1, 0.2, 0.2]])

                regression_diff = jt.abs(targets - regression[positive_indices, :])
                # smooth l1 loss
                regression_loss = jt.where(   regression_diff <= 1.0 / 9.0,
                                              0.5 * 9.0 * jt.pow(regression_diff, 2),
                                              regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(jt.array(0.0))

        return jt.stack(classification_losses).mean(dim=0, keepdim=True), \
               jt.stack(regression_losses).mean(dim=0, keepdim=True)


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = jt.minimum(jt.unsqueeze(a[:, 2], dim=1), b[:, 2]) - jt.maximum(jt.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = jt.minimum(jt.unsqueeze(a[:, 3], dim=1), b[:, 3]) - jt.maximum(jt.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = jt.clamp(iw, min_v=0)
    ih = jt.clamp(ih, min_v=0)

    ua = jt.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = jt.clamp(ua, min_v=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def test_focal_loss():
    import numpy as np
    jt.flags.use_cuda = False  # 若有 GPU 且想用 GPU，则设为 True

    # 参数设置
    batch_size       = 2
    num_anchors      = 5
    num_classes      = 3
    max_annotations  = 4

    # 随机生成“网络”输出
    # classifications: [B, A, C]，值在 (0,1) 之间
    classifications = jt.random([batch_size, num_anchors, num_classes])
    # regressions: [B, A, 4]，位置回归输出
    regressions     = jt.random([batch_size, num_anchors, 4])

    # 构造 anchors: [1, A, 4]，格式 [x1,y1,x2,y2]
    # 这里简单取均匀分布在 [0, 10] 上的坐标，并确保 x2 > x1, y2 > y1
    raw = jt.random([num_anchors, 4]) * 10
    x1 = jt.minimum(raw[:,0], raw[:,2])
    y1 = jt.minimum(raw[:,1], raw[:,3])
    x2 = jt.maximum(raw[:,0], raw[:,2])
    y2 = jt.maximum(raw[:,1], raw[:,3])
    anchors = jt.stack([x1, y1, x2, y2], dim=1).unsqueeze(0)

    # 构造 annotations: [B, M, 5]，最后一维 [x1,y1,x2,y2, class_id]
    annotations = jt.full([batch_size, max_annotations, 5], -1.0)
    for b in range(batch_size):
        # 随机给每张图生成 1 ~ max_annotations 个真值框
        n = np.random.randint(1, max_annotations+1)
        raw_gt = jt.random([n, 4]) * 10
        gx1 = jt.minimum(raw_gt[:,0], raw_gt[:,2])
        gy1 = jt.minimum(raw_gt[:,1], raw_gt[:,3])
        gx2 = jt.maximum(raw_gt[:,0], raw_gt[:,2])
        gy2 = jt.maximum(raw_gt[:,1], raw_gt[:,3])
        cls = jt.randint(0, num_classes, shape=[n,1]).float()
        gt = jt.concat([gx1.unsqueeze(1), gy1.unsqueeze(1),
                        gx2.unsqueeze(1), gy2.unsqueeze(1), cls], dim=1)
        annotations[b, :n, :] = gt

    # 实例化并运行 FocalLoss
    loss_fn = FocalLoss()
    cls_loss, reg_loss = loss_fn(classifications, regressions, anchors, annotations)

    print(f"分类损失 (cls_loss): {cls_loss.item():.6f}")
    print(f"回归损失 (reg_loss): {reg_loss.item():.6f}")

if __name__ == "__main__":
    test_focal_loss()