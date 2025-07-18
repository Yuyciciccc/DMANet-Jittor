import os
import numpy as np
from numpy.lib import recfunctions as rfn
import cv2
import numba as nb
from models.functions.voxel_generator import VoxelGenerator
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # close mulit-processing of open-cv
import jittor as jt
from jittor.dataset import Dataset

def getDataloader(name):
    dataset_dict = {"Prophesee": Prophesee}
    return dataset_dict.get(name)


class Prophesee(Dataset):
    def __init__(self, root, object_classes, height, width, mode="training",
                 voxel_size=None, max_num_points=None, max_voxels=None, resize=None, num_bins=None):
        """
        Creates an iterator over the Prophesee object recognition dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or "all" for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param mode: "training", "testing" or "validation"
        :param voxel_size: 
        :param max_num_points: 
        :param max_voxels: 
        :param num_bins: 
        """
        super(Prophesee, self).__init__()

        if mode == "training":
            mode = "train"
        elif mode == "validation":
            mode = "val"
        elif mode == "testing":
            mode = "test"

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height

        self.voxel_size = voxel_size
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.num_bins = num_bins

        self.voxel_generator = VoxelGenerator(voxel_size=self.voxel_size, point_cloud_range=[0, 0, 0, resize, resize, num_bins-1],
                                              max_num_points=self.max_num_points, max_voxels=self.max_voxels)
        self.resize = resize
        self.max_nr_bbox = 60

        filelist_path = os.path.join(self.root, self.mode)

        self.event_files, self.label_files, self.index_files = self.load_data_files(filelist_path, self.root, self.mode)

        assert len(self.event_files) == len(self.label_files)

        self.object_classes = object_classes
        self.nr_classes = len(self.object_classes)  # 7 classes

        self.nr_samples = len(self.event_files)
        # self.nr_samples = len(self.event_files) - len(self.index_files)*batch_size
        self.total_len = len(self.event_files)


    def __len__(self):
        return len(self.event_files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from split .npy files
        :param idx:
        :return: events: (x, y, t, p)
                boxes: (N, 4), which is consist of (x_min, y_min, x_max, y_max)
                histogram: (512, 512, 10)
        """
        # 初始化三个列表来保存该批次的标注、正样本和负样本
        boxes_list, pos_event_list, neg_event_list = [], [], []
        
        # 获取每一帧的标签文件和事件文件
        bbox_file = os.path.join(self.root, self.mode, "labels", self.label_files[idx])
        event_file = os.path.join(self.root, self.mode, "events", self.event_files[idx])

        # 加载标签和事件数据（标签是bounding boxes，事件是 (x, y, t, p)）
        labels_np = np.load(bbox_file)
        events_np = np.load(event_file)

        # 迭代处理每个标签文件中的数据
        for npz_num in range(len(labels_np)):
            # 创建一个大小为 max_nr_bbox 的空数组，用于保存标签
            const_size_box = np.ones([self.max_nr_bbox, 5]) * -1
            
            try:
                # 尝试读取该索引的标签和事件数据
                ev_npz = "e" + str(npz_num)
                lb_npz = "l" + str(npz_num)
                events_np_ = events_np[ev_npz]
                labels_np_ = labels_np[lb_npz]
            except:
                # 如果发生错误（如CRC错误），则尝试读取上一个索引的数据
                ev_npz = "e" + str(npz_num-1)
                lb_npz = "l" + str(npz_num-1)
                events_np_ = events_np[ev_npz]
                labels_np_ = labels_np[lb_npz]

            # 筛选掉不在帧内的事件（例如 x 或 y 超出了图像边界）
            mask = (events_np_['x'] < 1280) * (events_np_['y'] < 720)
            events_np_ = events_np_[mask]

            # 将结构化的标签和事件数据转为普通的 ndarray 格式
            labels = rfn.structured_to_unstructured(labels_np_)[:, [1, 2, 3, 4, 5]]  # (x, y, w, h, class_id)
            events = rfn.structured_to_unstructured(events_np_)[:, [1, 2, 0, 3]]  # (x, y, t, p)

            # 对标签进行裁剪，确保在图像内
            labels = self.cropToFrame(labels)
            labels = self.filter_boxes(labels, 60, 20)  # 过滤掉太小的框

            # 对事件流进行下采样，将分辨率从 1280x720 转为 512x512
            events = self.downsample_event_stream(events)

            # 将标签从 [x1, y1, x2, y2, class_id] 转化为标准化格式
            labels[:, 2] += labels[:, 0]
            labels[:, 3] += labels[:, 1]  # [x1, y1, x2, y2, class_id]
            labels[:, 0] /= 1280
            labels[:, 1] /= 720
            labels[:, 2] /= 1280
            labels[:, 3] /= 720

            # 将标签的范围缩放到 512x512 图像内
            labels[:, :4] *= 512
            labels[:, 2] -= labels[:, 0]
            labels[:, 3] -= labels[:, 1]
            labels[:, 2:-1] += labels[:, :2]  # [x_min, y_min, x_max, y_max, class_id]

            # 将事件根据正负标签进行分类
            pos_events = events[events[:, -1] == 1.0]  # positive events (p = 1)
            neg_events = events[events[:, -1] == 0.0]  # negative events (p = 0)
            pos_events = pos_events.astype(np.float32)
            neg_events = neg_events.astype(np.float32)
            
            # 如果没有负事件，使用正事件来替代
            if not len(neg_events):
                neg_events = pos_events
            if not len(pos_events):
                pos_events = neg_events

            # 使用体素生成器对事件数据进行处理，生成体素数据
            pos_voxels, pos_coordinates, pos_num_points = self.voxel_generator.generate(pos_events[:, :3], self.max_voxels)
            neg_voxels, neg_coordinates, neg_num_points = self.voxel_generator.generate(neg_events[:, :3], self.max_voxels)

            # 保存框和事件数据
            boxes = labels.astype(np.float32)
            const_size_box[:boxes.shape[0], :] = boxes
            boxes_list.append(const_size_box.astype(np.float32))
            pos_event_list.append([jt.array(pos_voxels), jt.array(pos_coordinates), jt.array(pos_num_points)])
            neg_event_list.append([jt.array(neg_voxels), jt.array(neg_coordinates), jt.array(neg_num_points)])

        boxes = np.array(boxes_list)  # 将所有的框合并成一个 numpy 数组
        # print("boxes shape:", boxes.shape)
        # print("pos_event_list shape:", len(pos_event_list), pos_event_list[0][0].shape, pos_event_list[0][1].shape, pos_event_list[0][2].shape)
        # print("neg_event_list shape:", len(neg_event_list), neg_event_list[0][0].shape, neg_event_list[0][1].shape, neg_event_list[0][2].shape)
        # labels, pos_events, neg_events = collate_events([
        #     (boxes[i], pos_event_list[i], neg_event_list[i]) for i in range(len(boxes))
        # ])
        
        # return labels, pos_events, neg_events
        return boxes , pos_event_list, neg_event_list

    def downsample_event_stream(self, events):
        events[:, 0] = events[:, 0] / 1280 * 512  # x
        events[:, 1] = events[:, 1] / 720 * 512  # y
        delta_t = events[-1, 2] - events[0, 2]
        events[:, 2] = 4 * (events[:, 2] - events[0, 2]) / delta_t

        _, ev_idx = np.unique(events[:, :2], axis=0, return_index=True)
        downsample_events = events[ev_idx]
        ev = downsample_events[np.argsort(downsample_events[:, 2])]
        return ev

    def normalize(self, histogram):
        """standard normalize"""
        nonzero_ev = (histogram != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = histogram.sum() / num_nonzeros
            stddev = np.sqrt((histogram ** 2).sum() / num_nonzeros - mean ** 2)
            histogram = nonzero_ev * (histogram - mean) / (stddev + 1e-8)
        return histogram

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        boxes = []
        for box in np_bbox:
            # if box[2] > 1280 or box[3] > 800:  # filter error label
            if box[2] > 1280:
                continue

            if box[0] < 0:  # x < 0 & w > 0
                box[2] += box[0]
                box[0] = 0
            if box[1] < 0:  # y < 0 & h > 0
                box[3] += box[1]
                box[1] = 0
            if box[0] + box[2] > self.width:  # x+w>1280
                box[2] = self.width - box[0]
            if box[1] + box[3] > self.height:  # y+h>720
                box[3] = self.height - box[1]

            if box[2] > 0 and box[3] > 0 and box[0] < self.width and box[1] <= self.height:
                boxes.append(box)
        boxes = np.array(boxes).reshape(-1, 5)
        return boxes

    def filter_boxes(self, boxes, min_box_diag=60, min_box_side=20):
        """Filters boxes according to the paper rule.
        To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
        To note: we assume the initial time of the video is always 0
        :param boxes: (np.ndarray)
                     structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence']
                     (example BBOX_DTYPE is provided in src/box_loading.py)
        Returns:
            boxes: filtered boxes
        """
        width = boxes[:, 2]
        height = boxes[:, 3]
        diag_square = width ** 2 + height ** 2
        mask = (diag_square >= min_box_diag ** 2) * (width >= min_box_side) * (height >= min_box_side)
        return boxes[mask]

    @staticmethod
    @nb.jit()
    def load_data_files(filelist_path, root, mode):
        idx = 0
        event_files = []
        label_files = []
        index_files = []
        filelist_dir = sorted(os.listdir(filelist_path))
        for filelist in filelist_dir:
            event_path = os.path.join(root, mode, filelist, "events")
            label_path = os.path.join(root, mode, filelist, "labels")
            data_dirs = sorted(os.listdir(event_path))

            for dirs in data_dirs:
                event_path_sub = os.path.join(event_path, dirs)
                label_path_sub = os.path.join(label_path, dirs)
                event_path_list = sorted(os.listdir(event_path_sub))
                label_path_list = sorted(os.listdir(label_path_sub))
                idx += len(event_path_list) - 1
                index_files.append(idx)

                for ev, lb in zip(event_path_list, label_path_list):
                    event_root = os.path.join(event_path_sub, ev)
                    label_root = os.path.join(label_path_sub, lb)
                    event_files.append(event_root)
                    label_files.append(label_root)
        return event_files, label_files, index_files

    def file_index(self):
        return self.index_files


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None):
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image, boxes



