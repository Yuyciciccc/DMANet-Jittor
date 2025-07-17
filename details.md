详细记录了如何将DMANet的pytorch版代码转化为jittor代码
只对重要部分进行介绍
参考：https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html

## cuda设置
### pytorch

pytorch为显式调用：每个 Tensor 或 Module 都要 .to(device)

### jittor

jittor采用全局开关，只需要在训练脚本前统一设置即可
修改为jittor版本只需要去除所有的显示调用，添加以下代码即可
``` train.py AbstractTrainer/__init__.py
    if settings.gpu_device != "cpu":
        jt.flags.use_cuda = 1
        jt.set_cuda_device(int(settings.gpu_device))
```

## models.functions

### anchors.py

numpy数组转换为jtensor
torch.from_numpy() -> jt.array()

### box_utils.py

torch.clamp(tensor , min = , max = ) -> jt.clamp(var , min_v = , max_v =)
torch.exp() -> jt.exp()
torch.stack() -> jt.stack()

! 逐元素比较大小 
torch.min() -> jt.minimum()
torch.max() -> jt.maximum()

### focal.py

torch.ones() -> jt.ones()
torch.log() -> jt.log()

! torch.max() 返回 (最大值,最大值索引)
jt.max()只返回最大值
jt.argmax()只返回最大值索引

jt没有查到类似于lt \ ge \ ne 等的比较函数，直接使用运算符即可
targets[torch.lt(IoU_max, 0.4), :] = 0 -> targets[IoU_max < 0.4, :] = 0
positive_indices = torch.ge(IoU_max, 0.5) -> positive_indices = IoU_max >= 0.5

torch.where() -> jt.where
targets.t() -> jt.transpose(targets)

### warmup.py

pytorch版本中使用了torch提供的_LRScheduler作为基类实现WarmUpLR，以实现动态调整学习率
在jittor中，同样提供了class jittor.optim.LRScheduler(optimizer, last_epoch=-1)
只需要进行简单替换即可


## models.modules

class 中forward 标识符直接替换为execute
import jittor.nn as nn

### embed_aggregator.py

nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2)   ->  nn.Conv(channels, channels, kernel_size, padding=(kernel_size-1)//2) or nn.Conv2(...)
在jittor中， nn.Conv 和 nn.Conv2d 完全相同
nn.ReLU(inplace=True) -> nn.relu() 在jittor中，inplace=True是默认的，且没有这个形参

### eventpillars.py

矩阵相乘
torch.mm() -> jittor.matual() or @

torch.nn.Linear()   -> jittor.nn.Linear()

torch.nn.BarchNorm2d() -> jittor.nn.BatchNorm() or jittor.nn.BatchNorm1d() or jittor.nn.BatchNorm2d() or jittor.nn.BatchNorm3d()

### dmanet_dector.py
torchvision.ops.nms(anchorBoxes, scores, iou_threshold)  -> jt.nms(dets, iou_threshold)
jittor的nms dets为[N,5],dets = [boxes, scores]
修改结果：
anchors_nms_idx = jt.nms(jt.concat([anchorBoxes, scores.unsqueeze(1)], dim=1) , self.iou_threshold)


### dmanet_network.py