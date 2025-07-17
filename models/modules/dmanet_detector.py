import jittor as jt
import jittor.nn as nn
from models.functions.box_utils import BBoxTransform, ClipBoxes


class DMANet_Detector(nn.Module):
    def __init__(self, conf_threshold, iou_threshold):
        super(DMANet_Detector, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def execute(self, classification, regression, anchors, img_batch):

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = jt.zeros((0,), dtype=jt.float32)

        finalAnchorBoxesIndexes = jt.zeros((0,), dtype=jt.int64)

        finalAnchorBoxesCoordinates = jt.zeros((0, 4), dtype=jt.float32)


        for i in range(classification.shape[2]):
            scores = jt.squeeze(classification[:, :, i])
            scores_over_thresh = (scores > self.conf_threshold)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = jt.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = jt.nms(jt.concat([anchorBoxes, scores.unsqueeze(1)], dim=1) , self.iou_threshold)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(jt.array([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = jt.cat((finalScores, scores[anchors_nms_idx]))
            
            finalAnchorBoxesIndexesValue = jt.array([i] * anchors_nms_idx.shape[0])


            finalAnchorBoxesIndexes = jt.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = jt.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        if len(finalScores):
            finalScores = finalScores.unsqueeze(-1)
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.type(jt.float32).unsqueeze(-1)
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates

            return jt.cat([finalAnchorBoxesCoordinates, finalScores, finalAnchorBoxesIndexes], dim=1)
        else:  # empty
            return jt.zeros((0, 6), dtype=jt.float32)

