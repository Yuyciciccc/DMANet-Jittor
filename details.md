# DMANet PyTorch 转 Jittor 代码指南
参考：https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html

## cuda设置
    pytorch为显式调用：每个 Tensor 或 Module 都要 .to(device)

    jittor采用全局开关，只需要在训练脚本前统一设置即可
    修改为jittor版本只需要去除所有的显示调用，添加以下代码即可
    ``` train_jittor.py AbstractTrainer/__init__.py
        if settings.gpu_device != "cpu":
            jt.flags.use_cuda = 1
            jt.set_cuda_device(int(settings.gpu_device))
    ```

## tensor -> var 操作转换
### 将numpy数组转换为对应框架的张量
    torch.from_numpy() -> jt.array()
### 元素裁剪
    torch.clamp(tensor , min = , max = ) -> jt.clamp(var , min_v = , max_v =)
### 指数运算
    torch.exp() -> jt.exp()
### 堆叠张量
    torch.stack() -> jt.stack()
### 逐元素比较大小
    torch.min(tensor,tensor) -> jt.minimum(var,var)
    torch.max(tensor,tensor) -> jt.maximum(var,var)
### 生成全为1的张量
    torch.ones() -> jt.ones()
### log
    torch.log() -> jt.log()
### 求最大值
    torch.max() 返回 max_val , max_val_index
    jt.max() 只返回 max_val

    torch.argmax() 只返回 max_val_index
    jt.argmax() 返回 max_val_index , max_val
### 比较函数
    torch.lt/torch.ne/...
    jittor中未发现相应的函数，但可直接使用运算符
    ```
    targets[torch.lt(IoU_max, 0.4), :] = 0 -> targets[IoU_max < 0.4, :] = 0
    positive_indices = torch.ge(IoU_max, 0.5) -> positive_indices = IoU_max >= 0.5
    ```
### 条件选择
    torch.where() -> jt.where()
### 转置
    tensor.t() -> jt.transpose(tensor)
### 类型转换
    tensor.type(tensor.float64) -> var.astype(jt.float64) 其他同理
### 矩阵相乘
    torch.mm() -> jt.matual() or @
    jittor中没有mm()
### size
    tensor.size()[0] -> jt.size(var)[0]
### nonzero
    torch.nonzero(as_tuple=False) -> jt.nonzero() 没有as_tuple参数
   
## 学习率调度
主要涉及文件：
    models/functions/warmup.py
    train_jittor.py
本项目采用两阶段学习率策略：
### warmup 预热
    将pytorch版本中WarmUpLR类继承的父类_LRScheduler替换为jittor.optim.LRScheduler
### 余弦退火策略
    在jittor框架中提供了CosineAnnealingLR类，但未完善相关功能，故在warmup.py中手动实现了CosineAnnealingLR类
    具体代码可以查看models/functions/warmup.py

## 模型转换
### nn 
    import torch.nn as nn ->import jittor.nn as nn 
### forward
    def forward(self, ... ) -> def execute(self, ... )
### conv
    nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2)   ->  nn.Conv(channels, channels, kernel_size, padding=(kernel_size-1)//2) or nn.Conv2(...)
    在jittor中， nn.Conv 和 nn.Conv2d 完全相同
### relu
    nn.ReLU(inplace=True) -> nn.Relu() or nn.ReLU() 在jittor中，inplace=True是默认的，且没有这个形参
### linear
    nn.Linear()   -> nn.Linear()
### batchnorm
    nn.BarchNorm2d() -> nn.BatchNorm() or nn.BatchNorm1d() or nn.BatchNorm2d() or nn.BatchNorm3d()
### 正态分布初始化权重 
    nn.init.normal_(module.weight, mean, std) -> nn.init.gauss_(module.weight,mean,std)
### nms
    torchvision.ops.nms(anchorBoxes, scores, iou_threshold)  -> jt.nms(dets, iou_threshold)
    jittor的nms dets为[N,5],dets = [boxes, scores]
    ```示例  dmanet_dector.py
    anchors_nms_idx = jt.nms(jt.concat([anchorBoxes, scores.unsqueeze(1)], dim=1) , self.iou_threshold)
    ```
### 读取模型参数(dmanet_network.py)
    Pytorch版本下 使用model_zoo.load_url读取resnet模型参数
    在jittor版本中，我使用自定义的load_pretrained_resnet_weights函数读取参数
### SubsetRandomSampler
    self.train_sampler = jt.dataset.SubsetRandomSampler((train_dataset),(0 , len(train_dataset) - 1))
    self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(len(train_dataset))))
### dataset dataloader/
    删去dataloader中的pin_memory参数
    jittor.dataset.Dataset 的collate_fn参数更名为collate_batch，
### 反向传播示例
    下面三种均可：
    ``` 1 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```
    ``` 2
    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()
    ```
    ``` 3
    optimizer.step(loss)
    ```
### 参数梯度访问方式
    p.grad()  -> p.opt_grad()
### 参数裁切正则化
    pytorch : nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
    jittor : self.optimizer.clip_grad_norm(0.1)  
### 读取保存模型
    jittor 版本只需简单操作
    model.save()
    model.load()
