详细记录了如何将DMANet的pytorch版代码转化为jittor代码
只对重要部分进行介绍
参考：https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html

tensor.type(tensor.float64) -> var.astype(jt.float64) 其他同理
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

numpy转换为tensor/Var
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
jt.argmax()返回最大值索引,最大值

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
nn.ReLU(inplace=True) -> nn.Relu() or nn.ReLU() 在jittor中，inplace=True是默认的，且没有这个形参

### eventpillars.py

矩阵相乘
torch.mm() -> jittor.matual() or @

torch.nn.Linear()   -> jittor.nn.Linear()

torch.nn.BarchNorm2d() -> jittor.nn.BatchNorm() or jittor.nn.BatchNorm1d() or jittor.nn.BatchNorm2d() or jittor.nn.BatchNorm3d()


### non_local_aggregator.py
    正态分布初始化权重
    nn.init.normal_(module.weight, mean, std)
-> nn.init.gauss_(module.weight,mean,std)
### dmanet_dector.py
torchvision.ops.nms(anchorBoxes, scores, iou_threshold)  -> jt.nms(dets, iou_threshold)
jittor的nms dets为[N,5],dets = [boxes, scores]
修改结果：
anchors_nms_idx = jt.nms(jt.concat([anchorBoxes, scores.unsqueeze(1)], dim=1) , self.iou_threshold)

### convlstm_fusion.py

batch_size = input_.data.size()[0]
spatial_size = input_.data.size()[2:]

batch_size = jt.size(input_.data)[0]
spatial_size = jt.size(input_.data)[2:]


### dmanet_network.py

Pytorch版本下 使用model_zoo.load_url读取resnet模型参数
在jittor版本中，我使用load_pretrained_resnet_weights函数读取参数
模型的参数使用jittor官方的库为resnet.resnet18(pretrained=True)

## train-jittor.py

        self.train_sampler = jt.dataset.SubsetRandomSampler((train_dataset),(0 , len(train_dataset) - 1))
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(len(train_dataset))))
## dataloader/dataset.py

在jittor中添加total_len属性   self.total_len = len(self.data)  
删去pin_memory 
jittor.dataset.Dataset 的collate_fn参数更名为collate_batch，


Jittor 的 getitem 运算（即 Var[...]）目前仅支持单一维度或等价于 ellipsis 的简单切片，不支持复杂的多维切片。
因此，我们需要使用更通用的函数接口：jittor.misc.index_select(...)。

PyTorch:   tensor[:, i]         
Jittor:    index_select(tensor, 1, jt.array([i])).squeeze(1)

For example, if your code looks like this::

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

It can be changed to this::

    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()

Or more concise::

    optimizer.step(loss)

The step function will automatically zero grad and backward.

参数梯度访问方式
p.grad()  -> p.opt_grad()

nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
self.optimizer.clip_grad_norm(0.1)  

model.save()
model.load()


torch.nonzero(as_tuple=False):
-> jt.nonzero() 没有as_tuple参数
 
