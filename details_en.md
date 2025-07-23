# DMANet PyTorch to Jittor Code Guide
Reference: https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html

## CUDA Setup
    pytorch requires explicit calls: each Tensor or Module must .to(device)

    jittor uses a global switch; set once at the start of the training script
    To convert to jittor, remove all explicit calls and add:
    ``` train_jittor.py AbstractTrainer/__init__.py
        if settings.gpu_device != "cpu":
            jt.flags.use_cuda = 1
            jt.set_cuda_device(int(settings.gpu_device))
    ```

## Tensor → Var Operation Mappings
### Convert numpy array to framework tensor
    torch.from_numpy() -> jt.array()
### Clamp values
    torch.clamp(tensor, min=…, max=…) -> jt.clamp(var, min_v=…, max_v=…)
### Exponential
    torch.exp() -> jt.exp()
### Stack tensors
    torch.stack() -> jt.stack()
### Element-wise min/max
    torch.min(tensor, tensor) -> jt.minimum(var, var)
    torch.max(tensor, tensor) -> jt.maximum(var, var)
### Create all-ones tensor
    torch.ones() -> jt.ones()
### Logarithm
    torch.log() -> jt.log()
### Max value
    torch.max() returns max_val, max_idx
    jt.max() returns only max_val

    torch.argmax() returns only max_idx
    jt.argmax() returns max_idx, max_val
### Comparison functions
    torch.lt/torch.ne/...
    Jittor has no direct equivalents; use operators:
    ```
    targets[torch.lt(IoU_max, 0.4), :] = 0 -> targets[IoU_max < 0.4, :] = 0
    positive_indices = torch.ge(IoU_max, 0.5) -> positive_indices = IoU_max >= 0.5
    ```
### Conditional selection
    torch.where() -> jt.where()
### Transpose
    tensor.t() -> jt.transpose(tensor)
### Type casting
    tensor.type(torch.float64) -> var.astype(jt.float64) (others similar)
### Matrix multiplication
    torch.mm() -> jt.matmul() or @
    Jittor has no mm()
### Size
    tensor.size()[0] -> jt.size(var)[0]
### Non-zero indices
    torch.nonzero(as_tuple=False) -> jt.nonzero() (no as_tuple parameter)

## Learning Rate Scheduling
Main files:
    models/functions/warmup.py
    train_jittor.py
This project uses a two-stage LR strategy:
### Warm-up
    Replace PyTorch WarmUpLR base class (_LRScheduler) with jittor.optim.LRScheduler
### Cosine annealing
    Jittor provides CosineAnnealingLR but it's incomplete, so a custom version is implemented in warmup.py (see models/functions/warmup.py)

## Model Conversion
### nn import
    import torch.nn as nn -> import jittor.nn as nn
### forward method
    def forward(self, ... ) -> def execute(self, ... )
### conv
    nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2)
    -> nn.Conv(channels, channels, kernel_size, padding=(kernel_size-1)//2) or nn.Conv2(...)
    (nn.Conv and nn.Conv2d are identical in Jittor)
### relu
    nn.ReLU(inplace=True) -> nn.Relu() or nn.ReLU() (inplace=True is default and not available)
### linear
    nn.Linear() -> nn.Linear()
### batchnorm
    nn.BatchNorm2d() -> nn.BatchNorm() or nn.BatchNorm1d() or nn.BatchNorm2d() or nn.BatchNorm3d()
### Gaussian weight initialization
    nn.init.normal_(module.weight, mean, std) -> nn.init.gauss_(module.weight, mean, std)
### nms
    torchvision.ops.nms(anchorBoxes, scores, iou_threshold)
    -> jt.nms(dets, iou_threshold)
    Jittor’s nms expects dets as [N,5] combining [boxes, scores]
    ``` example in dmanet_detector.py
    anchors_nms_idx = jt.nms(
        jt.concat([anchorBoxes, scores.unsqueeze(1)], dim=1),
        self.iou_threshold
    )
    ```
### Loading model parameters (dmanet_network.py)
    PyTorch: model_zoo.load_url for resnet weights
    Jittor: custom load_pretrained_resnet_weights function
### SubsetRandomSampler
    jt.dataset.SubsetRandomSampler(train_dataset, (0, len(train_dataset)-1))
    -> torch.utils.data.sampler.SubsetRandomSampler(list(range(len(train_dataset))))
### Dataset & DataLoader
    Remove pin_memory from DataLoader
    jittor.dataset.Dataset renamed collate_fn to collate_batch
### Backpropagation examples
    All three work:
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
### Gradient access
    p.grad -> p.opt_grad()
### Gradient clipping
    pytorch: nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    jittor: optimizer.clip_grad_norm(0.1)
### Saving & loading models
    model.save()
    model.load()
