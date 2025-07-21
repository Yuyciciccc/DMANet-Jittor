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

import logging
import time

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
jt.cudnn.set_max_workspace_ratio(0.0)

class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings
        self._init_logger()
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

    def _init_logger(self):
        """Create file logger and dump all hyperparameters."""

        self.logger = logging.getLogger("ExperimentLogger")
        self.logger.setLevel(logging.DEBUG)
        os.makedirs(self.settings.log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(self.settings.log_dir, f"train.log")
        )
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info("===== Experiment Configuration =====")
        for k, v in vars(self.settings).items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("====================================")

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

        # Jittor中的数据采样器
        self.train_sampler = jt.dataset.SubsetRandomSampler(train_dataset,(0 , len(train_dataset) - 1))

        self.train_loader = self.dataset_loader(train_dataset, mode="training",
                                                batch_size=self.settings.batch_size,
                                                num_workers=self.settings.num_cpu_workers,
                                                drop_last=True, sampler=self.train_sampler,
                                                data_index=self.train_file_indexes)

        self.val_loader = self.dataset_loader(val_dataset, mode="validation",
                                              batch_size=self.settings.batch_size ,
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
        self.model.save(file_path)

class DMANetDetection(AbstractTrainer):
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = actual_num.unsqueeze(axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = jt.arange(max_num, dtype=jt.int32).view(max_num_shape)
        paddings_indicator = actual_num.int32() > max_num
        return paddings_indicator

    def process_pillar_input(self, events, idx, idy):
        try:
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
        print("\033[0;33m Using pretrained model! \033[0m")
        # checkpoint = jt.load(self.settings.pretrained_model)
        # try:
        #     pretrained_dict = checkpoint["state_dict"]
        # except KeyError:
        #     pretrained_dict = checkpoint["model"]
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if "input_layer." not in k}
        # self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.load(self.settings.pretrained_model)

    def train(self):
        while self.epoch_step <= self.settings.epoch:
            self.trainEpoch()
            self.validationEpoch()
            self.epoch_step += 1
        
    def trainEpoch(self):
        # 初始化日志记录器
        if not hasattr(self, 'train_logger'):
            self.log_dir = os.path.join(self.settings.log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 创建文本日志记录器
            self.train_logger = logging.getLogger('train_logger')
            self.train_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(os.path.join(self.log_dir, "training_log.txt"))
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.train_logger.addHandler(fh)
            
            # 创建CSV文件头
            with open(os.path.join(self.log_dir, "timing_stats.csv"), 'w') as f:
                f.write("epoch,batch,data_time,fwd_time,bwd_time,total_time,cls_loss,reg_loss,total_loss\n")
            
            # 创建epoch摘要文件头
            with open(os.path.join(self.log_dir, "epoch_summary.csv"), 'w') as f:
                f.write("epoch,total_time,avg_data_time,avg_fwd_time,avg_bwd_time,avg_batch_time\n")
            
            self.epoch_times = []
            self.batch_times = []
            self.data_times = []
            self.fwd_times = []
            self.bwd_times = []

        epoch_start = time.time()
        last_batch_end = epoch_start
        batch_times = []
        epoch_data_time = 0
        epoch_fwd_time = 0
        epoch_bwd_time = 0

        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit="Batch", desc=f"Epoch {self.epoch_step}")
        self.model.train()
        focal_loss = FocalLoss()    

        for i_batch, sample_batched in enumerate(self.train_loader):
            batch_start = time.time()
            
            load_time = batch_start - last_batch_end
            epoch_data_time += load_time
            self.data_times.append(load_time)
            
            if self.epoch_step < self.settings.warm:
                self.warmup_schedular.step()

            bounding_box, pos_events, neg_events = sample_batched
            prev_states, prev_features = None, None

            # 初始化损失变量
            batch_total_loss = jt.array(0.0)
            batch_cls_loss = jt.array(0.0)
            batch_reg_loss = jt.array(0.0)
            loss_count = 0

            fwd_start = time.time()
            for idx in range(self.settings.seq_len):
                pos_input_list, neg_input_list = [], []
                bounding_box_now, bounding_box_next = [], []

                for idy in range(self.settings.batch_size):
                    pos_input = self.process_pillar_input(pos_events, idx, idy)
                    neg_input = self.process_pillar_input(neg_events, idx, idy)
                    pos_input_list.append(pos_input)
                    neg_input_list.append(neg_input)

                    mask_now = (bounding_box[:, -1] == (idy * self.settings.seq_len + idx))
                    mask_next = (bounding_box[:, -1] == (idy * self.settings.seq_len + idx + 1))
                    bbox_now = bounding_box[mask_now][:, :-1]
                    bbox_next = bounding_box[mask_next][:, :-1]
                    bounding_box_now.append(bbox_now)
                    bounding_box_next.append(bbox_next)

                classification, regression, anchors, prev_states, prev_features, _ = \
                    self.model([pos_input_list, neg_input_list], prev_states=prev_states, prev_features=prev_features)

                if isinstance(bounding_box_now, list):
                    bboxes_tensor = jt.array(bounding_box_now)
                else:
                    bboxes_tensor = bounding_box_now

                cls_loss, reg_loss = focal_loss(classification, regression, anchors, bboxes_tensor)
                cls_loss_mean = cls_loss.mean()
                reg_loss_mean = reg_loss.mean()
                step_loss = cls_loss_mean + reg_loss_mean

                batch_total_loss += step_loss
                batch_cls_loss += cls_loss_mean
                batch_reg_loss += reg_loss_mean
                loss_count += 1
            fwd_time = time.time() - fwd_start
            epoch_fwd_time += fwd_time
            self.fwd_times.append(fwd_time)

            if loss_count > 0:
                batch_total_loss = batch_total_loss / loss_count
                batch_cls_loss = batch_cls_loss / loss_count
                batch_reg_loss = batch_reg_loss / loss_count

            bwd_start = time.time()
            self.optimizer.step(batch_total_loss)
            self.optimizer.clip_grad_norm(0.1)
            if self.epoch_step >= self.settings.warm:
                self.train_schedular.step()
            bwd_time = time.time() - bwd_start
            epoch_bwd_time += bwd_time
            self.bwd_times.append(bwd_time)

            cls_loss_val = batch_cls_loss.item() if hasattr(batch_cls_loss, 'item') else float(batch_cls_loss)
            reg_loss_val = batch_reg_loss.item() if hasattr(batch_reg_loss, 'item') else float(batch_reg_loss)
            total_loss_val = batch_total_loss.item() if hasattr(batch_total_loss, 'item') else float(batch_total_loss)

            last_batch_end = time.time()
            batch_dur = time.time() - batch_start
            batch_times.append((i_batch, batch_dur))
            self.batch_times.append(batch_dur)

            # 记录批处理日志
            log_msg = (
                f"Epoch {self.epoch_step} Batch {i_batch} - "
                f"Data: {load_time:.4f}s | "
                f"Fwd: {fwd_time:.4f}s | "
                f"Bwd: {bwd_time:.4f}s | "
                f"Total: {batch_dur:.4f}s | "
                f"Cls Loss: {cls_loss_val:.4f} | "
                f"Reg Loss: {reg_loss_val:.4f} | "
                f"Total Loss: {total_loss_val:.4f}"
            )
            self.train_logger.info(log_msg)
            
            # 写入CSV
            with open(os.path.join(self.log_dir, "timing_stats.csv"), 'a') as f:
                f.write(f"{self.epoch_step},{i_batch},{load_time:.6f},{fwd_time:.6f},{bwd_time:.6f},{batch_dur:.6f},{cls_loss_val:.6f},{reg_loss_val:.6f},{total_loss_val:.6f}\n")
            
            # 更新进度条
            self.pbar.set_postfix(
                Conf=cls_loss_val, Loc=reg_loss_val, Total=total_loss_val,
                Load=f"{load_time:.3f}s", Fwd=f"{fwd_time:.3f}s", Bwd=f"{bwd_time:.3f}s"
            )

            loss_list = [
                jt.array(cls_loss_val),
                jt.array(reg_loss_val),
                jt.array(total_loss_val)
            ]
            self.storeLossesObjectDetection(loss_list)
            self.pbar.update(1)
            self.batch_step += 1

            self.writer.add_scalar("Training/Learning_Rate_batch", self.getLearningRate(), self.batch_step)

            jt.sync_all()
            jt.gc()
            jt.display_memory_info()

            if i_batch % self.settings.log_interval == 0:
                self.logger.debug(
                    f"[Train][E{self.epoch_step}][B{i_batch}/{len(self.train_loader)}] "
                    f"Loss={total_loss_val:.4f}  LR={self.getLearningRate():.6f}  "
                    f"Load={load_time:.3f}s  Fwd={fwd_time:.3f}s  Bwd={bwd_time:.3f}s"
                )

        self.pbar.close()
        epoch_dur = time.time() - epoch_start
        self.epoch_times.append(epoch_dur)
        
        # 计算并记录平均时间
        avg_data_time = epoch_data_time / len(self.train_loader)
        avg_fwd_time = epoch_fwd_time / len(self.train_loader)
        avg_bwd_time = epoch_bwd_time / len(self.train_loader)
        avg_batch_time = sum(self.batch_times[-len(self.train_loader):]) / len(self.train_loader)
        
        epoch_log = (
            f"Epoch {self.epoch_step} Summary - "
            f"Total: {epoch_dur:.2f}s | "
            f"Avg Data: {avg_data_time:.4f}s | "
            f"Avg Fwd: {avg_fwd_time:.4f}s | "
            f"Avg Bwd: {avg_bwd_time:.4f}s | "
            f"Avg Batch: {avg_batch_time:.4f}s"
        )
        self.train_logger.info(epoch_log)
        self.train_logger.info("-" * 80)
        
        # 写入CSV总结
        with open(os.path.join(self.log_dir, "epoch_summary.csv"), 'a') as f:
            f.write(f"{self.epoch_step},{epoch_dur:.4f},{avg_data_time:.4f},{avg_fwd_time:.4f},{avg_bwd_time:.4f},{avg_batch_time:.4f}\n")
        
        avg_batch = sum(t for _, t in batch_times) / len(batch_times)
        self.logger.info(
            f"[Train][Epoch {self.epoch_step} Done] "
            f"Time={epoch_dur:.1f}s  AvgBatch={avg_batch:.3f}s  EndLR={self.getLearningRate():.6f}"
        )
        self.writer.add_scalar("Training/Learning_Rate", self.getLearningRate(), self.epoch_step)

        if self.epoch_step % 1 == 0:
            self.saveCheckpoint()

    def validationEpoch(self):
        # 初始化验证日志
        if not hasattr(self, 'val_logger'):
            self.val_logger = logging.getLogger('val_logger')
            self.val_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(os.path.join(self.log_dir, "validation_log.txt"))
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.val_logger.addHandler(fh)
            
            # 创建验证时间统计CSV
            with open(os.path.join(self.log_dir, "val_timing.csv"), 'w') as f:
                f.write("epoch,batch,fwd_time,total_time\n")
            self.val_fwd_times = []
            self.val_times = []
        
        val_start = time.time()
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit="Batch", unit_scale=True, desc="Validation")

        self.model.eval()
        dmanet_detector = DMANet_Detector(conf_threshold=0.1, iou_threshold=0.5)

        iouv = jt.linspace(0.5, 0.95, 10)
        niou = iouv.numel()

        seen = 0
        stats = []
        scale = jt.array([self.settings.resize] * 4, dtype=jt.float32)
        prev_states = None
        
        # 用于记录验证时间
        total_val_fwd_time = 0
        val_batch_count = 0

        for i_batch, sample_batched in enumerate(self.val_loader):
            
            t0 = time.time()
            bounding_box, pos_events, neg_events = sample_batched
            detection_result = []
            prev_features = None
            batch_fwd_time = 0

            with jt.no_grad():
                for idx in range(self.settings.seq_len):
                    pos_input_list, neg_input_list = [], []
                    for idy in range(self.settings.batch_size):
                        pos_input = self.process_pillar_input(pos_events, idx, idy)
                        neg_input = self.process_pillar_input(neg_events, idx, idy)
                        # print(f"pos_input: {pos_input}")
                        # print(f"neg_input: {neg_input}")
                        pos_input_list.append(pos_input), neg_input_list.append(neg_input)
                    fwd_start = time.time()
                    classification, regression, anchors, prev_states, prev_features, pseudo_img = \
                        self.model([pos_input_list, neg_input_list], prev_states, prev_features)
                    fwd_time = time.time() - fwd_start
                    batch_fwd_time += fwd_time
                    
                    out = dmanet_detector(classification, regression, anchors, pseudo_img)
                    detection_result.append(out)
                    jt.sync_all()
                    jt.gc()

            total_val_fwd_time += batch_fwd_time
            self.val_fwd_times.append(batch_fwd_time)
            
            batch_time = time.time() - t0
            self.val_times.append(batch_time)
            
            # 记录批处理日志
            self.val_logger.info(f"Validation Epoch {self.epoch_step} Batch {i_batch} - Fwd: {batch_fwd_time:.4f}s | Total: {batch_time:.4f}s")
            
            # 写入CSV
            with open(os.path.join(self.log_dir, "val_timing.csv"), 'a') as f:
                f.write(f"{self.epoch_step},{i_batch},{batch_fwd_time:.6f},{batch_time:.6f}\n")

            self.pbar.update(1)
            val_batch_count += 1
            print(len(detection_result))
            for si, pred in enumerate(detection_result):
                bbox = bounding_box[bounding_box[:, -1] == si]
                np_labels = bbox[bbox[:, -2] != -1.][:, [4,0,1,2,3]]  # [cls, x1, y1, x2, y2]
                labels = jt.array(np_labels)
                nl = len(labels)
                tcls = labels[:, 0].numpy().tolist() if nl else []
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((jt.zeros(0, niou, dtype=jt.bool), jt.array([]), jt.array([]), tcls))
                    continue

                predn = pred.clone()
                predn[:, :4] /= self.settings.resize
                predn[:, :4] *= scale
                correct = jt.zeros(pred.shape[0], niou, dtype=jt.bool)

                if nl:
                    tcls_tensor = labels[:, 0]
                    tbox = labels[:, 1:5]
                    for cls in jt.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero().view(-1)
                        pi = (cls == pred[:, 5]).nonzero().view(-1)
                        
                        if pi.numel() == 0 or ti.numel() == 0:
                            continue

                        try:
                            result = box_iou(predn[pi, :4], tbox[ti])
                            i,ious = jt.argmax(result, dim = 1)

                        except Exception as e:
                            self.logger.warning(f"[Validation] box_iou failed: {e}")
                            continue
                        
                        detected = [] 
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero():
                            # print(j, i, ti, iouv, ious)
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in images
                                    break

                stats.append((correct.numpy(), pred[:, 4].numpy(), pred[:, 5].numpy(), tcls))

            if i_batch % self.settings.log_interval == 0:
                self.logger.debug(
                    f"[Val][E{self.epoch_step}][B{i_batch}/{len(self.val_loader)}] "
                    f"Time={batch_time:.3f}s"
                )

        self.pbar.close()
        val_dur = time.time() - val_start
        
        # 记录验证总结
        avg_val_fwd = total_val_fwd_time / val_batch_count if val_batch_count > 0 else 0
        self.val_logger.info(f"Validation Summary Epoch {self.epoch_step} - Total: {val_dur:.2f}s | Avg Fwd: {avg_val_fwd:.4f}s")
        self.val_logger.info("-" * 80)

        # --- 评估指标 ---
        stats_np = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []
        names = {k: v for k, v in enumerate(self.object_classes)}
        if stats_np and stats_np[0].any():
            precision, recall, ap, _, ap_class = ap_per_class(*stats_np, plot=False, save_dir=os.path.join(self.settings.save_dir, "det_result"), names=names)
            ap50, ap = ap[:, 0], ap.mean(axis=1)
            m_precision, m_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
        else:
            m_precision = m_recall = map50 = map = 0.0

        self.logger.info(f"[Validation][E{self.epoch_step}] TotalTime={val_dur:.1f}s")
        self.logger.info(f"[Metrics] mAP@0.5={map50:.3f}  mAP@0.5:0.95={map:.3f}  P={m_precision:.3f}  R={m_recall:.3f}")

        self.writer.add_scalar("Validation/mAP@0.5", map50, self.epoch_step)
        self.writer.add_scalar("Validation/mAP@0.5:0.95", map, self.epoch_step)
        self.writer.add_scalar("Validation/Precision", m_precision, self.epoch_step)
        self.writer.add_scalar("Validation/Recall", m_recall, self.epoch_step)

    def test(self):
        # 初始化测试日志
        if not hasattr(self, 'test_logger'):
            self.test_logger = logging.getLogger('test_logger')
            self.test_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(os.path.join(self.log_dir, "test_log.txt"))
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.test_logger.addHandler(fh)
            
            # 创建测试时间统计CSV
            with open(os.path.join(self.log_dir, "test_timing.csv"), 'w') as f:
                f.write("sequence,data_time,fwd_time,total_time\n")
            self.test_fwd_times = []
            self.test_data_times = []
            self.test_times = []
        
        # 创建测试数据集
        test_dataset = self.dataset_builder(self.settings.dataset_path,
                                            self.settings.object_classes,
                                            self.settings.height,
                                            self.settings.width,
                                            mode="testing",
                                            voxel_size=self.settings.voxel_size,
                                            max_num_points=self.settings.max_num_points,
                                            max_voxels=self.settings.max_voxels,
                                            resize=self.settings.resize,
                                            num_bins=self.settings.num_bins)
        
        # 创建测试数据加载器
        test_loader = self.dataset_loader(test_dataset, mode="testing",
                                            batch_size=self.settings.batch_size,
                                            num_workers=self.settings.num_cpu_workers,
                                            pin_memory=False,
                                            drop_last=False,
                                            sampler=None,
                                            data_index=test_dataset.file_index())
        
        print(f"Starting testing on {len(test_loader)} sequences...")
        self.model.eval()
        dmanet_detector = DMANet_Detector(conf_threshold=0.1, iou_threshold=0.5)
        
        # 创建结果保存目录
        results_dir = os.path.join(self.settings.save_dir, "test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # 进度条
        pbar = tqdm.tqdm(total=len(test_loader), unit="Sequence", desc="Testing")
        
        # 用于计算mAP的统计信息
        stats = []
        seen = 0
        iouv = jt.linspace(0.5, 0.95, 10)
        niou = iouv.numel()
        scale = jt.array([self.settings.resize] * 4, dtype=jt.float32)
        
        test_start = time.time()
        with jt.no_grad():
            for seq_idx, sample_batched in enumerate(test_loader):
                seq_start = time.time()
                # 初始化状态
                prev_states, prev_features = None, None
                all_detections = []
                
                bounding_box, pos_events, neg_events = sample_batched
                
                # 记录数据加载时间
                data_time = time.time() - seq_start
                self.test_data_times.append(data_time)
                
                # 确定实际的序列长度
                actual_seq_len = self.settings.seq_len
                if pos_events and len(pos_events) > 0:
                    actual_seq_len = min(self.settings.seq_len, len(pos_events[0]))
                
                # 确定实际的批次大小
                actual_batch_size = self.settings.batch_size
                if pos_events and len(pos_events) > 0:
                    actual_batch_size = min(self.settings.batch_size, len(pos_events))
                
                seq_fwd_time = 0
                
                # 处理序列中的每个时间步
                for t in range(actual_seq_len):
                    pos_input_list, neg_input_list = [], []

                    for batch_idx in range(actual_batch_size):
                        try:
                            # 添加边界检查
                            if (batch_idx < len(pos_events) and 
                                t < len(pos_events[batch_idx]) and
                                batch_idx < len(neg_events) and 
                                t < len(neg_events[batch_idx])):
                                
                                # 处理正负事件
                                pos_input = self.process_pillar_input(pos_events, t, batch_idx)
                                neg_input = self.process_pillar_input(neg_events, t, batch_idx)
                                
                                pos_input_list.append(pos_input)
                                neg_input_list.append(neg_input)
                            else:
                                # 如果索引超出范围，跳过这个批次
                                print(f"Warning: Index out of range at seq {seq_idx}, batch {batch_idx}, time {t}")
                                print(f"  pos_events length: {len(pos_events)}")
                                print(f"  pos_events[{batch_idx}] length: {len(pos_events[batch_idx]) if batch_idx < len(pos_events) else 'N/A'}")
                                continue
                        except Exception as e:
                            print(f"Error processing batch {batch_idx} at time {t}: {str(e)}")
                            continue
                    
                    # 如果没有有效的输入，跳过这个时间步
                    if not pos_input_list or not neg_input_list:
                        print(f"Warning: No valid inputs at time {t}, skipping...")
                        continue
                    
                    try:
                        # 记录前向时间
                        fwd_start = time.time()
                        # 模型推理
                        classification, regression, anchors, prev_states, prev_features, pseudo_img = \
                            self.model([pos_input_list, neg_input_list], 
                                        prev_states=prev_states, 
                                        prev_features=prev_features)
                        fwd_time = time.time() - fwd_start
                        seq_fwd_time += fwd_time
                        self.test_fwd_times.append(fwd_time)
                        
                        # 检测器处理输出
                        detections = dmanet_detector(classification, regression, anchors, pseudo_img)
                        all_detections.append(detections)
                    except Exception as e:
                        print(f"Error in model inference at time {t}: {str(e)}")
                        all_detections.append([])  # 添加空检测结果
                        continue
                
                # 保存当前序列的检测结果
                self.save_sequence_results(seq_idx, all_detections, results_dir)
                
                # 如果有标签则计算mAP
                if bounding_box is not None and len(all_detections) > 0:
                    # 处理每个时间步的预测和标签
                    for t, detections in enumerate(all_detections):
                        try:
                            # 获取当前时间步的标签
                            bbox = bounding_box[bounding_box[:, -1] == t]
                            if bbox.size == 0:
                                continue
                                
                            labels = bbox[bbox[:, -2] != -1][:, [4, 0, 1, 2, 3]]  # [cls, coords]
                            labels = jt.array(labels)
                            
                            nl = len(labels)
                            tcls = labels[:, 0].numpy().tolist() if nl else []
                            seen += 1
                            
                            if len(detections) == 0:
                                if nl:
                                    stats.append((jt.zeros(0, niou, dtype=jt.bool), 
                                                    jt.array([]), jt.array([]), tcls))
                                continue
                            
                            # 处理预测结果
                            pred = detections.clone()
                            pred[:, :4] /= self.settings.resize  # 归一化坐标
                            pred[:, :4] *= scale  # 恢复到原始分辨率
                            
                            # 初始化正确性矩阵
                            correct = jt.zeros(pred.shape[0], niou, dtype=jt.bool)
                            
                            if nl:
                                detected = []
                                tcls_tensor = labels[:, 0]
                                tbox = labels[:, 1:5]
                                
                                # 对每个类别进行匹配
                                for cls in jt.unique(tcls_tensor):
                                    ti = (cls == tcls_tensor).nonzero().view(-1)
                                    pi = (cls == pred[:, 5]).nonzero().view(-1)
                                    
                                    if pi.numel() > 0 and ti.numel() > 0:
                                        # 计算IoU
                                        result = box_iou(pred[pi, :4], tbox[ti])
                                        ious = jt.max(result, dim = 1)
                                        i = jt.argmax(result, dim = 1)
                                        # 标记正确检测
                                        detected_set = set()
                                        for j in (ious > iouv[0]).nonzero().view(-1):
                                            d = ti[i[j]]
                                            if d.item() not in detected_set:
                                                detected_set.add(d.item())
                                                detected.append(d)
                                                correct[pi[j]] = ious[j] > iouv
                                                if len(detected) == nl:
                                                    break
                            
                            stats.append((correct.numpy(), pred[:, 4].numpy(), pred[:, 5].numpy(), tcls))
                        except Exception as e:
                            print(f"Error processing mAP calculation at time {t}: {str(e)}")
                            continue
                
                # 记录序列时间
                seq_time = time.time() - seq_start
                self.test_times.append(seq_time)
                self.test_logger.info(f"Sequence {seq_idx} - Data: {data_time:.4f}s | Fwd: {seq_fwd_time:.4f}s | Total: {seq_time:.4f}s")
                
                # 写入CSV
                with open(os.path.join(self.log_dir, "test_timing.csv"), 'a') as f:
                    f.write(f"{seq_idx},{data_time:.6f},{seq_fwd_time:.6f},{seq_time:.6f}\n")
                
                pbar.update(1)
        
        pbar.close()
        
        # 如果有标签，计算并打印mAP
        if stats:
            print("\nCalculating mAP...")
            try:
                stats_np = [np.concatenate(x, 0) for x in zip(*stats)]
                
                if stats_np and stats_np[0].any():
                    precision, recall, ap, f1, ap_class = ap_per_class(*stats_np, plot=True, save_dir=os.path.join(results_dir, 'img'))
                    ap50, ap = ap[:, 0], ap.mean(axis=1)
                    mp, mr, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
                    
                    # 打印总体结果
                    header_pf = "%8s" + "%18s" * 2 + "%19s" * 4
                    data_pf   = "%8s" + "%18i" * 2 + "%19.5f" * 4

                    # 打印表头
                    print(header_pf % ("Class", "Images", "Labels", "Precision", "Recall", "mAP@.5", "mAP@.5:.95"))
                    # 打印总体数据
                    print(data_pf   % ("all", seen, len(stats_np[3]), mp, mr, map50, map))

                    # 打印每个类别结果
                    for i, c in enumerate(ap_class):
                        print(data_pf % (
                            self.object_classes[c], seen,
                            (stats_np[3] == c).sum(),
                            precision[i], recall[i], ap50[i], ap[i]
                        ))

                    # 保存结果到文件
                    with open(os.path.join(results_dir, "map_results.txt"), "w") as f:
                        # 表头
                        f.write(header_pf % ("Class", "Images", "Labels", "Precision", "Recall", "mAP@.5", "mAP@.5:.95") + "\n")
                        # 总体
                        f.write(data_pf   % ("all", seen, len(stats_np[3]), mp, mr, map50, map) + "\n")
                        # 每类
                        for i, c in enumerate(ap_class):
                            f.write(data_pf % (
                                self.object_classes[c], seen,
                                (stats_np[3] == c).sum(),
                                precision[i], recall[i], ap50[i], ap[i]
                            ) + "\n")

            except Exception as e:
                print(f"Error calculating mAP: {str(e)}")
        
        # 记录测试总结
        test_time = time.time() - test_start
        avg_test_data = sum(self.test_data_times) / len(self.test_data_times) if self.test_data_times else 0
        avg_test_fwd = sum(self.test_fwd_times) / len(self.test_fwd_times) if self.test_fwd_times else 0
        self.test_logger.info(f"Test Summary - Total: {test_time:.2f}s | Avg Data: {avg_test_data:.4f}s | Avg Fwd: {avg_test_fwd:.4f}s")
        self.test_logger.info("-" * 80)
        
        print(f"Testing completed. Results saved to {results_dir}")

    def save_sequence_results(self, seq_idx, detections, results_dir):
        """保存序列的检测结果到文件"""
        seq_dir = os.path.join(results_dir, f"seq_{seq_idx:04d}")
        os.makedirs(seq_dir, exist_ok=True)
        
        for t, frame_detections in enumerate(detections):
            # 如果没有检测到目标，创建空文件
            if frame_detections is None or len(frame_detections) == 0:
                with open(os.path.join(seq_dir, f"frame_{t:04d}.txt"), 'w') as f:
                    pass  # 创建空文件
                continue
            
            # 保存检测结果：class_id, conf, x, y, w, h
            try:
                with open(os.path.join(seq_dir, f"frame_{t:04d}.txt"), 'w') as f:
                    for det in frame_detections:
                        cls_id = int(det[5])
                        conf = det[4]
                        x, y, w, h = det[:4]
                        
                        # 保存为: class confidence x y w h
                        f.write(f"{cls_id} {conf:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}\n")
            except Exception as e:
                print(f"Error saving results for seq {seq_idx}, frame {t}: {str(e)}")
                # 创建空文件作为fallback
                with open(os.path.join(seq_dir, f"frame_{t:04d}.txt"), 'w') as f:
                    pass


def main():
    """Main function to run the training"""
    parser = argparse.ArgumentParser(description='Train DMANet')
    parser.add_argument('--settings_file', type=str, default='config/settings.yaml',
                        help='Path to the settings file')
    args = parser.parse_args()

    settings = Settings(args.settings_file)
    trainer = DMANetDetection(settings)
    trainer.train()

if __name__ == '__main__':
    main()


