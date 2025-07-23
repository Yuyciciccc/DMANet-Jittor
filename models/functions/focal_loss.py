import jittor as jt
import jittor.nn as nn


def calc_iou(a, b):
    """
    Compute pairwise IoU between anchors a [A,4] and boxes b [M,4], returning [A,M].
    """
    # Intersection width and height
    iw = jt.minimum(jt.unsqueeze(a[:, 2], 1), b[:, 2]) - \
         jt.maximum(jt.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = jt.minimum(jt.unsqueeze(a[:, 3], 1), b[:, 3]) - \
         jt.maximum(jt.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = jt.clamp(iw, min_v=0)
    ih = jt.clamp(ih, min_v=0)

    # Intersection area
    inter = iw * ih
    # Union area
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # [A]
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # [M]
    union = jt.unsqueeze(area_a, 1) + area_b - inter
    union = jt.clamp(union, min_v=1e-8)
    return inter / union


class FocalLoss(nn.Module):
    def execute(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size, num_anchors, num_classes = classifications.shape

        cls_losses = []
        reg_losses = []

        # anchors: [1,A,4] -> anchor: [A,4]
        anchor = anchors[0]
        aw = anchor[:, 2] - anchor[:, 0]
        ah = anchor[:, 3] - anchor[:, 1]
        acx = anchor[:, 0] + 0.5 * aw
        acy = anchor[:, 1] + 0.5 * ah

        for b in range(batch_size):
            # clamp for numerical stability
            cls_pred = jt.clamp(classifications[b], min_v=1e-4, max_v=1.0-1e-4)  # [A,C]
            reg_pred = regressions[b]                                             # [A,4]

            # filter annotations
            ann = annotations[b]
            ann = ann[ann[:, 4] != -1]

            # no ground truths => all negatives
            if ann.numel() == 0:
                af = (1 - alpha) * jt.ones_like(cls_pred)
                fw = af * jt.pow(cls_pred, gamma)
                bce = -jt.log(1 - cls_pred)
                cls_losses.append((fw * bce).sum())
                reg_losses.append(jt.zeros(1))
                continue

            # compute IoU between anchors and GT boxes
            iou = calc_iou(anchor, ann[:, :4])  # [A,M]
            iou_argmax, iou_max = jt.argmax(iou, dim=1)

            # prepare classification targets
            targets = jt.full_like(cls_pred, -1.0)
            neg_mask = iou_max < 0.4
            pos_mask = iou_max >= 0.5
            neg_idx = neg_mask.nonzero().view(-1)   # [N]
            pos_idx = pos_mask.nonzero().view(-1)   # [P]

            # set negatives to 0
            targets[neg_idx, :] = 0

            # assign positives
            assigned = ann[iou_argmax]             # [A,5]
            targets[pos_idx, :] = 0
            cls_idx = assigned[pos_idx, 4].int32()
            targets[pos_idx, cls_idx] = 1

            num_pos = pos_idx.numel()

            # classification focal loss
            af = jt.where(targets == 1, alpha, 1 - alpha)
            fw = jt.where(targets == 1, 1 - cls_pred, cls_pred)
            fw = af * jt.pow(fw, gamma)
            bce = -(targets * jt.log(cls_pred) + (1 - targets) * jt.log(1 - cls_pred))
            loss_cls = jt.where(targets != -1, fw * bce, jt.zeros_like(bce))
            cls_losses.append(loss_cls.sum() / jt.clamp(jt.array(num_pos), min_v=1.0))

            # regression loss on positives
            if num_pos > 0:
                # select only positive anchors
                pos_assigned = jt.index_select(assigned, 0, pos_idx)  # [P,5]
                aw_p = aw[pos_idx]                                   # [P]
                ah_p = ah[pos_idx]                                   # [P]
                acx_p = acx[pos_idx]                                 # [P]
                acy_p = acy[pos_idx]                                 # [P]

                # compute GT deltas for positives
                gt_w = pos_assigned[:, 2] - pos_assigned[:, 0]
                gt_h = pos_assigned[:, 3] - pos_assigned[:, 1]
                gt_cx = pos_assigned[:, 0] + 0.5 * gt_w
                gt_cy = pos_assigned[:, 1] + 0.5 * gt_h
                gt_w = jt.clamp(gt_w, min_v=1)
                gt_h = jt.clamp(gt_h, min_v=1)

                dx = (gt_cx - acx_p) / aw_p
                dy = (gt_cy - acy_p) / ah_p
                dw = jt.log(gt_w / aw_p)
                dh = jt.log(gt_h / ah_p)

                reg_targets = jt.stack([dx, dy, dw, dh], dim=1) / jt.array([0.1, 0.1, 0.2, 0.2])  # [P,4]

                # select corresponding predictions
                pred_pos = jt.index_select(reg_pred, 0, pos_idx)  # [P,4]
                diff = jt.abs(reg_targets - pred_pos)
                reg_loss = jt.where(diff <= 1/9,
                                    0.5 * 9 * diff * diff,
                                    diff - 0.5/9)
                reg_losses.append(reg_loss.mean())
            else:
                reg_losses.append(jt.zeros(1))

        return jt.stack(cls_losses).mean(dim=0, keepdim=True), jt.stack(reg_losses).mean(dim=0, keepdim=True)


def test_focal_loss():
    import numpy as np
    jt.flags.use_cuda = False

    B, A, C, M = 2, 5, 3, 4
    cls_pred = jt.random([B, A, C])
    reg_pred = jt.random([B, A, 4])

    raw = jt.random([A, 4]) * 10
    x1 = jt.minimum(raw[:, 0], raw[:, 2])
    y1 = jt.minimum(raw[:, 1], raw[:, 3])
    x2 = jt.maximum(raw[:, 0], raw[:, 2])
    y2 = jt.maximum(raw[:, 1], raw[:, 3])
    anchors = jt.stack([x1, y1, x2, y2], dim=1).unsqueeze(0)

    ann = jt.full([B, M, 5], -1.0)
    for b in range(B):
        n = np.random.randint(1, M+1)
        gt = jt.random([n, 5]) * 10
        gt[:, 2:4] = jt.maximum(gt[:, 0:2], gt[:, 2:4])
        gt[:, 4] = jt.randint(0, C, shape=[n])
        ann[b, :n, :] = gt

    loss_fn = FocalLoss()
    cls_l, reg_l = loss_fn(cls_pred, reg_pred, anchors, ann)
    print(f"cls_loss={cls_l.item():.4f}, reg_loss={reg_l.item():.4f}")
