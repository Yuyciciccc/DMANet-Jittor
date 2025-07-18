import jittor as jt
import jittor.nn as nn


class FocalLoss(nn.Module):
    def execute(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        num_anchors = classifications.shape[1]
        num_classes = classifications.shape[2]

        classification_losses = []
        regression_losses = []

        anchor = anchors[0]  # [A,4]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for b in range(batch_size):
            classification = jt.clamp(classifications[b], 1e-4, 1.0 - 1e-4)  # [A,C]
            regression     = regressions[b]                                  # [A,4]
            bbox_annotation = annotations[b]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.numel() == 0:
                alpha_factor = (1 - alpha) * jt.ones_like(classification)
                focal_weight = alpha_factor * jt.pow(classification, gamma)
                bce = -jt.log(1.0 - classification)
                classification_losses.append((focal_weight * bce).sum())
                regression_losses.append(jt.zeros(1))
                continue

            IoU = calc_iou(anchor, bbox_annotation[:, :4])   # [A, M]
            IoU_max = jt.max(IoU, dim=1)                     # [A]
            IoU_argmax = jt.argmax(IoU, dim=1)               # [A]

            targets = jt.full(classification.shape, -1.0)    # [A,C]

            # —— 用 index_select 替换原来的布尔切片 —— #

            # Negatives: IoU < 0.4 → 0
            neg_mask = IoU_max < 0.4
            neg_rows = jt.index_select(jt.arange(num_anchors), 0, neg_mask.nonzero()[0])
            targets[neg_rows, :] = 0

            # Positives: IoU ≥ 0.5 → one‑hot
            pos_mask = IoU_max >= 0.5
            assigned_annotations = bbox_annotation[IoU_argmax]    # [A,5]
            pos_rows = jt.index_select(jt.arange(num_anchors), 0, pos_mask.nonzero()[0])
            # 1) 清零正样本行
            targets[pos_rows, :] = 0
            # 2) 扁平索引打 one‑hot
            sel_rows = jt.index_select(assigned_annotations, 0, pos_rows)  # [P,5]
            cls_idx = jt.index_select(sel_rows, 1, jt.array([4])).squeeze(1).int32()  # shape: [P]  # [P]
            A = num_anchors
            C = num_classes
            flat_idx = pos_rows * C + cls_idx                             # [P]
            targets_flat = targets.reshape([-1])                           # [A*C]
            targets_flat[flat_idx] = 1
            targets = targets_flat.reshape([A, C])


            num_pos = len(pos_rows)

            # classification loss
            alpha_factor = jt.where(targets == 1, alpha, 1 - alpha)
            focal_weight = jt.where(targets == 1, 1 - classification, classification)
            focal_weight = alpha_factor * jt.pow(focal_weight, gamma)
            bce = -(targets * jt.log(classification) +
                    (1 - targets) * jt.log(1 - classification))
            cls_loss = jt.where(targets != -1,
                                focal_weight * bce,
                                jt.zeros_like(bce))
            classification_losses.append(cls_loss.sum() /
                                         jt.clamp(jt.array(num_pos), min_v=1.0))

            # regression loss
            if num_pos > 0:
                pos_assigned = sel_rows
                aw = anchor_widths[pos_rows]
                ah = anchor_heights[pos_rows]
                acx = anchor_ctr_x[pos_rows]
                acy = anchor_ctr_y[pos_rows]

                x1 = jt.index_select(pos_assigned, 1, jt.array([0])).squeeze(1)
                y1 = jt.index_select(pos_assigned, 1, jt.array([1])).squeeze(1)
                x2 = jt.index_select(pos_assigned, 1, jt.array([2])).squeeze(1)
                y2 = jt.index_select(pos_assigned, 1, jt.array([3])).squeeze(1)

                gw = x2 - x1
                gh = y2 - y1
                
                x1 = jt.index_select(pos_assigned, 1, jt.array([0])).squeeze(1)
                y1 = jt.index_select(pos_assigned, 1, jt.array([1])).squeeze(1)

                gcx = x1 + 0.5 * gw
                gcy = y1 + 0.5 * gh
                gw = jt.clamp(gw, min_v=1)
                gh = jt.clamp(gh, min_v=1)

                dx = (gcx - acx) / aw
                dy = (gcy - acy) / ah
                dw = jt.log(gw / aw)
                dh = jt.log(gh / ah)

                reg_targets = jt.stack([dx, dy, dw, dh],
                                       dim=1) / jt.array([0.1, 0.1, 0.2, 0.2])
                diff = jt.abs(reg_targets - regression[pos_rows])
                reg_loss = jt.where(diff <= 1/9,
                                    0.5 * 9 * diff * diff,
                                    diff - 0.5/9)
                regression_losses.append(reg_loss.mean())
            else:
                regression_losses.append(jt.zeros(1))

        return jt.stack(classification_losses).mean(), \
               jt.stack(regression_losses).mean()


def calc_iou(a, b):
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = jt.minimum(jt.unsqueeze(a[:, 2], 1), b[:, 2]) - \
         jt.maximum(jt.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = jt.minimum(jt.unsqueeze(a[:, 3], 1), b[:, 3]) - \
         jt.maximum(jt.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = jt.clamp(iw, min_v=0)
    ih = jt.clamp(ih, min_v=0)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ua = jt.unsqueeze(ua, 1) + area_b - iw * ih
    ua = jt.clamp(ua, min_v=1e-8)
    return iw * ih / ua


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