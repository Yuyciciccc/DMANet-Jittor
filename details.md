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

## models


## functions

### warmup.py

pytorch版本中使用了torch提供的_LRScheduler作为基类实现WarmUpLR，以实现动态调整学习率
在jittor中，同样提供了class jittor.optim.LRScheduler(optimizer, last_epoch=-1)
只需要进行简单替换即可

### 