"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train_DMANet.py --settings_file "config/settings.yaml"
"""
import argparse
import os
import abc
import tqdm
import jittor as jt
import math
import numpy as np
import jittor.nn as nn
import jittor.optim as optim
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import dataloader.dataset
from models.functions.focal_loss import FocalLoss
from models.functions.box_utils import box_iou
from models.modules import dmanet_network
from models.modules.dmanet_detector import DMANet_Detector
from dataloader.loader import Loader
from models.functions.smooth_l1_loss import Smooth_L1_Loss
from config.settings import Settings
from utils.metrics import ap_per_class
from models.functions.warmup import WarmUpLR

# 设置Jittor使用GPU
jt.flags.use_cuda = 1


class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.scheduler = None
        self.nr_classes = None   # numbers of classes
        self.train_loader = None
        self.val_loader = None
        self.nr_train_epochs = None
        self.nr_val_epochs = None
        self.train_file_indexes = None
        self.val_file_indexes = None
        self.object_classes = None
        self.train_sampler = None

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)  # Prophesee
        self.dataset_loader = Loader
        self.writer = SummaryWriter(self.settings.ckpt_dir)

        self.createDatasets()  # train_dataset and val_dataset

        self.buildModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.settings.init_lr)

        # Jittor中的学习率调度器
        self.warmup_schedular = WarmUpLR(self.optimizer, len(self.train_loader)*self.settings.warm)
        self.train_schedular = jt.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                 T_max=len(self.train_loader)*(self.settings.epoch-self.settings.warm),
                                                                 eta_min=self.settings.init_lr*0.1)

        self.batch_step = 0
        self.epoch_step = 0
        self.training_loss = 0
        self.training_accuracy = 0
        self.max_validation_mAP = 0
        self.softmax = nn.Softmax(dim=-1)
        self.smooth_l1_loss = Smooth_L1_Loss(beta=0.11, reduction="sum")

        # tqdm progress bar
        self.pbar = None

        if settings.resume_training:
            self.loadCheckpoint(self.settings.resume_ckpt_file)

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = self.dataset_builder(self.settings.dataset_path,
                                             self.settings.object_classes,
                                             self.settings.height,
                                             self.settings.width,
                                             mode="training",
                                             voxel_size=self.settings.voxel_size,
                                             max_num_points=self.settings.max_num_points,
                                             max_voxels=self.settings.max_voxels,
                                             resize=self.settings.resize,
                                             num_bins=self.settings.num_bins)
        self.train_file_indexes = train_dataset.file_index()
        self.nr_train_epochs = train_dataset.nr_samples // self.settings.batch_size + 1
        self.nr_classes = train_dataset.nr_classes
        self.object_classes = train_dataset.object_classes

        val_dataset = self.dataset_builder(self.settings.dataset_path,
                                           self.settings.object_classes,
                                           self.settings.height,
                                           self.settings.width,
                                           mode="validation",
                                           voxel_size=self.settings.voxel_size,
                                           max_num_points=self.settings.max_num_points,
                                           max_voxels=self.settings.max_voxels,
                                           resize=self.settings.resize,
                                           num_bins=self.settings.num_bins)
        self.val_file_indexes = val_dataset.file_index()
        self.nr_val_epochs = val_dataset.nr_samples
        # print(f"Length of train_dataset: {len(train_dataset)}")

        # Jittor中的数据采样器
        self.train_sampler = jt.dataset.SubsetRandomSampler(train_dataset,(0 , len(train_dataset) - 1))

        self.train_loader = self.dataset_loader(train_dataset, mode="training",
                                                batch_size=self.settings.batch_size,
                                                num_workers=self.settings.num_cpu_workers,
                                                drop_last=True, sampler=self.train_sampler,
                                                data_index=self.train_file_indexes)

        self.val_loader = self.dataset_loader(val_dataset, mode="validation",
                                              batch_size=self.settings.batch_size // self.settings.batch_size,
                                              num_workers=self.settings.num_cpu_workers,
                                              drop_last=True, sampler=None,
                                              data_index=self.val_file_indexes)

    def storeLossesObjectDetection(self, loss_list):
        """Writes the different losses to tensorboard"""
        loss_names = ["Confidence_Loss", "Location_Loss", "Overall_Loss"]

        for idx in range(len(loss_list)):
            loss_value = loss_list[idx].data
            if hasattr(loss_value, 'numpy'):
                loss_value = loss_value.numpy()
            elif hasattr(loss_value, 'item'):
                loss_value = loss_value.item()
            self.writer.add_scalar("TrainingLoss/" + loss_names[idx], loss_value, self.batch_step)

    def storeClassmAP(self, map_list):
        class_names = self.settings.object_classes
        for idx in range(len(class_names)):
            class_map = map_list[idx]
            class_name = class_names[idx] + "_mAP"
            self.writer.add_scalar("Validation/"+class_name, class_map, self.epoch_step)
        self.writer.add_scalar("Validation/mAP0.5", map_list.mean(), self.epoch_step)

    def getLearningRate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def loadCheckpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = jt.load(filename)
            self.epoch_step = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def saveCheckpoint(self):
        file_path = os.path.join(self.settings.ckpt_dir, "model_step_" + str(self.epoch_step) + ".pth")
        jt.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "epoch": self.epoch_step}, file_path)


class DMANetDetection(AbstractTrainer):
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Create boolean mask by actually number of a padded tensor.
        :param actual_num:
        :param max_num:
        :param axis:
        :return: [type]: [description]
        """
        actual_num = actual_num.unsqueeze(axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = jt.arange(max_num, dtype=jt.int32).view(max_num_shape)
        paddings_indicator = actual_num.int32() > max_num
        # paddings_indicator shape : [batch_size, max_num]
        return paddings_indicator

    def process_pillar_input(self, events, idx, idy):
        try:
            # 添加边界检查
            if idy >= len(events):
                raise IndexError(f"idy ({idy}) >= len(events) ({len(events)})")
            
            if idx >= len(events[idy]):
                raise IndexError(f"idx ({idx}) >= len(events[{idy}]) ({len(events[idy])})")
            
            if len(events[idy][idx]) == 0:
                raise IndexError(f"events[{idy}][{idx}] is empty")
            
            pillar_x = events[idy][idx][0][..., 0].unsqueeze(0).unsqueeze(0)
            pillar_y = events[idy][idx][0][..., 1].unsqueeze(0).unsqueeze(0)
            pillar_t = events[idy][idx][0][..., 2].unsqueeze(0).unsqueeze(0)
            coors = events[idy][idx][1]
            num_points_per_pillar = events[idy][idx][2].unsqueeze(0)
            num_points_per_a_pillar = pillar_x.size()[3]
            mask = self.get_paddings_indicator(num_points_per_pillar, num_points_per_a_pillar, axis=0)
            mask = mask.permute(0, 2, 1).unsqueeze(1).float()
            
            # 在Jittor中直接使用tensor，不需要.cuda()
            input = [pillar_x, pillar_y, pillar_t, num_points_per_pillar, mask, coors]
            return input
        except Exception as e:
            print(f"Error in process_pillar_input: {str(e)}")
            print(f"  idy: {idy}, idx: {idx}")
            print(f"  len(events): {len(events)}")
            if idy < len(events):
                print(f"  len(events[{idy}]): {len(events[idy])}")
                if idx < len(events[idy]):
                    print(f"  len(events[{idy}][{idx}]): {len(events[idy][idx])}")
            raise

    def buildModel(self):
        """Creates the specified model"""
        if self.settings.depth == 18:
            self.model = dmanet_network.DMANet18(in_channels=self.settings.nr_input_channels,
                                                num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 34:
            self.model = dmanet_network.DMANet34(in_channels=self.settings.nr_input_channels,
                                                num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 50:
            self.model = dmanet_network.DMANet50(in_channels=self.settings.nr_input_channels,
                                                num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 101:
            self.model = dmanet_network.DMANet101(in_channels=self.settings.nr_input_channels,
                                                 num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 152:
            self.model = dmanet_network.DMANet152(in_channels=self.settings.nr_input_channels,
                                                 num_classes=len(self.settings.object_classes), pretrained=False)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        self.params_initialize()
        
        if self.settings.use_pretrained:
            self.loadPretrainedWeights()

    def params_initialize(self):
        print("\033[0;33m Starting to initialize parameters! \033[0m")
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                jt.init.gauss_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)
        
        prior = 0.01
        jt.init.constant_(self.model.classificationModel.output.weight, 0)
        jt.init.constant_(self.model.classificationModel.output.bias, -math.log((1.0 - prior) / prior))
        jt.init.constant_(self.model.regressionModel.output.weight, 0)
        jt.init.constant_(self.model.regressionModel.output.bias, 0)

    def loadPretrainedWeights(self):
        """Loads pretrained model weights"""
        print("\033[0;33m Using pretrained model! \033[0m")
        checkpoint = jt.load(self.settings.pretrained_model)
        try:
            pretrained_dict = checkpoint["state_dict"]
        except KeyError:
            pretrained_dict = checkpoint["model"]

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "input_layer." not in k}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def train(self):
        """Main training and validation loop"""
        while self.epoch_step <= self.settings.epoch:
            self.trainEpoch()
            self.validationEpoch()
            self.epoch_step += 1

    def trainEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit="Batch", unit_scale=True,
                            desc="Epoch: {}".format(self.epoch_step))
        self.model.train()
        focal_loss = FocalLoss()    

        for i_batch, sample_batched in enumerate(self.train_loader):
            if self.epoch_step < self.settings.warm:
                self.warmup_schedular.step()
            
            bounding_box, pos_events, neg_events = sample_batched
            prev_states, prev_features = None, None
            
            # 初始化损失变量
            batch_total_loss = jt.array(0.0)
            batch_cls_loss = jt.array(0.0)
            batch_reg_loss = jt.array(0.0)
            
            loss_count = 0

            for idx in range(self.settings.seq_len):
                pos_input_list, neg_input_list = [], []
                bounding_box_now, bounding_box_next = [], []
                
                for idy in range(self.settings.batch_size):
                    # 处理正负事件
                    pos_input = self.process_pillar_input(pos_events, idx, idy)
                    neg_input = self.process_pillar_input(neg_events, idx, idy)
                    pos_input_list.append(pos_input)
                    neg_input_list.append(neg_input)

                    # 标签处理
                    mask_now = (bounding_box[:, -1] == (idy * self.settings.seq_len + idx))
                    mask_next = (bounding_box[:, -1] == (idy * self.settings.seq_len + idx + 1))
                    bbox_now = bounding_box[mask_now][:, :-1]
                    bbox_next = bounding_box[mask_next][:, :-1]
                    bounding_box_now.append(bbox_now)
                    bounding_box_next.append(bbox_next)

                classification, regression, anchors, prev_states, prev_features, _ = \
                    self.model([pos_input_list, neg_input_list], 
                            prev_states=prev_states, 
                            prev_features=prev_features)
                
                # 转换边界框为Jittor张量
                if isinstance(bounding_box_now, list):
                    bboxes_tensor = jt.array(bounding_box_now)
                else:
                    bboxes_tensor = bounding_box_now
                
                cls_loss, reg_loss = focal_loss(classification, regression, 
                                            anchors, bboxes_tensor)
                
                # 计算平均损失
                cls_loss_mean = cls_loss.mean()
                reg_loss_mean = reg_loss.mean()
                step_loss = cls_loss_mean + reg_loss_mean
                
                # 累积损失
                batch_total_loss += step_loss
                batch_cls_loss += cls_loss_mean
                batch_reg_loss += reg_loss_mean
                loss_count += 1

            # 计算平均损失
            if loss_count > 0:
                batch_total_loss = batch_total_loss / loss_count
                batch_cls_loss = batch_cls_loss / loss_count
                batch_reg_loss = batch_reg_loss / loss_count

            # 反向传播
            self.optimizer.step(batch_total_loss)
            self.optimizer.clip_grad_norm(0.1)  
            
            # 获取损失值用于记录
            cls_loss_val = batch_cls_loss.item() if hasattr(batch_cls_loss, 'item') else float(batch_cls_loss)
            reg_loss_val = batch_reg_loss.item() if hasattr(batch_reg_loss, 'item') else float(batch_reg_loss)
            total_loss_val = batch_total_loss.item() if hasattr(batch_total_loss, 'item') else float(batch_total_loss)
            
            # 更新进度条和记录损失
            self.pbar.set_postfix(Conf=cls_loss_val, Loc=reg_loss_val, Total_Loss=total_loss_val)
            loss_list = [
                jt.array(cls_loss_val), 
                jt.array(reg_loss_val), 
                jt.array(total_loss_val)
            ]
            self.storeLossesObjectDetection(loss_list)
            self.pbar.update(1)
            self.batch_step += 1
            
            # 记录批次学习率
            self.writer.add_scalar("Training/Learning_Rate_batch", self.getLearningRate(), self.batch_step)

            # 学习率调度
            if self.epoch_step >= self.settings.warm:
                self.train_schedular.step()
        
        # 记录epoch学习率
        self.writer.add_scalar("Training/Learning_Rate", self.getLearningRate(), self.epoch_step)
        self.pbar.close()

    def validationEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit="Batch", unit_scale=True)
        self.model.eval()
        dmanet_detector = DMANet_Detector(conf_threshold=0.1, iou_threshold=0.5)

        iouv = jt.linspace(0.5, 0.95, 10)
        niou = iouv.numel()  # len(iouv)
        seen = 0
        precision, recall, f1_score, m_precision, m_recall, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []

        scale = jt.array([self.settings.resize, self.settings.resize, self.settings.resize, self.settings.resize],
                        dtype=jt.float32)

        prev_states = None
        for i_batch, sample_batched in enumerate(self.val_loader):
            prev_features = None
            detection_result = []  # detection result for computing mAP
            bounding_box, pos_events, neg_events = sample_batched

            with jt.no_grad():
                for idx in range(self.settings.seq_len):
                    pos_input_list, neg_input_list = [], []
                    for idy in range(self.settings.batch_size):
                        # process positive/negative events
                        pos_input = self.process_pillar_input(pos_events, idx, idy)
                        neg_input = self.process_pillar_input(neg_events, idx, idy)

                        pos_input_list.append(pos_input), neg_input_list.append(neg_input)
                    classification, regression, anchors, prev_states, prev_features, pseudo_img = \
                        self.model([pos_input_list, neg_input_list], prev_states=prev_states, prev_features=prev_features)
                    # [coords, scores, labels]
                    out = dmanet_detector(classification, regression, anchors, pseudo_img)
                    detection_result.append(out)
            self.pbar.update(1)

            for si, pred in enumerate(detection_result):
                bbox = bounding_box[bounding_box[:, -1] == si]  # each batch
                np_labels = bbox[bbox[:, -2] != -1.]
                np_labels = np_labels[:, [4, 0, 1, 2, 3]]  # [cls, coords]
                labels = jt.array(np_labels)

                nl = len(labels)
                tcls = labels[:, 0].numpy().tolist() if nl else []
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((jt.zeros(0, niou, dtype=jt.bool), jt.array([]), jt.array([]), tcls))
                    continue
                
                # predictions
                predn = pred.clone()
                predn[:, :4] /= self.settings.resize  # percent coordinates
                predn[:, :4] *= scale  # absolute coordinates, 1280x720

                # assign all predictions as incorrect
                correct = jt.zeros(pred.shape[0], niou, dtype=jt.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]
                    # target boxes
                    tbox = labels[:, 1:5]
                    # per target class
                    for cls in jt.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices
                        # search for detections
                        if pi.shape[0]:
                            # prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero():
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in images
                                        break
                stats.append((correct.numpy(), pred[:, 4].numpy(), pred[:, 5].numpy(), tcls))
        self.pbar.close()

        # Directories
        save_dir = os.path.join(self.settings.save_dir, "det_result")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)   # make dir
        names = {k: v for k, v in enumerate(self.object_classes, start=0)}  # {0: 'Pedestrian', ...}
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            precision, recall, ap, f1_score, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(axis=1)
            m_precision, m_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nr_classes-1)  # number of targets per class
        else:
            nt = jt.zeros(1)

        # Print results
        pf = "%8s" + "%18i" * 2 + "%19.3g" * 4  # print format
        print("\033[0;31m    Class            Events              Labels           Precision           Recall          "
              "   mAP@0.5           mAP@0.5:0.95 \033[0m")
        print(pf % ("all", seen, nt.sum(), m_precision, m_recall, map50, map))

        # Print results per class
        if len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], precision[i], recall[i], ap50[i], ap[i]))
                self.writer.add_scalar("Validation/" + names[c], ap50[i], self.epoch_step)
        self.writer.add_scalar("Validation/mAP@0.5", map50, self.epoch_step)
        
        if self.epoch_step % 5 == 0:
            self.saveCheckpoint()


def main():
    """Main function to run the training"""
    parser = argparse.ArgumentParser(description='Train DMANet')
    parser.add_argument('--settings_file', type=str, default='config/settings.yaml',
                        help='Path to the settings file')
    args = parser.parse_args()
    
    # Load settings
    settings = Settings(args.settings_file)
    
    # Create trainer and start training
    trainer = DMANetDetection(settings)
    trainer.train()



if __name__ == '__main__':
    main()