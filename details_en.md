## DMANet PyTorch → Jittor Migration Guide

This guide focuses on the essential steps to port the DMANet implementation from PyTorch to Jittor. For full API references, please consult the [Jittor documentation](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html).

---

### 1. Global CUDA Configuration

* **PyTorch** requires explicit `.to(device)` calls on every tensor or module.
* **Jittor** uses a global switch. Remove all explicit `.to()` calls and add at the start of your training script:

  ```python
  # train_jittor.py or AbstractTrainer/__init__.py
  import jittor as jt
  from config import settings  # assume settings.gpu_device is defined

  if settings.gpu_device != "cpu":
      jt.flags.use_cuda = 1
      jt.set_cuda_device(int(settings.gpu_device))
  ```

---

### 2. Core Tensor/Var Operation Mappings

| Operation             | PyTorch API                            | Jittor API                            |
| --------------------- | -------------------------------------- | ------------------------------------- |
| From NumPy            | `torch.from_numpy(ndarray)`            | `jt.array(ndarray)`                   |
| Clamp values          | `torch.clamp(x, min=, max=)`           | `jt.clamp(x, min_v=, max_v=)`         |
| Exponential           | `torch.exp(x)`                         | `jt.exp(x)`                           |
| Stack                 | `torch.stack(list)`                    | `jt.stack(list)`                      |
| Elementwise min/max   | `torch.min(a,b)` / `torch.max(a,b)`    | `jt.minimum(a,b)` / `jt.maximum(a,b)` |
| All-ones tensor       | `torch.ones(shape)`                    | `jt.ones(shape)`                      |
| Logarithm             | `torch.log(x)`                         | `jt.log(x)`                           |
| Max value & indices   | `torch.max(x, dim)` → `(values, idxs)` | `jt.max(x, dim)` → `values` only      |
| Argmax                | `torch.argmax(x, dim)` → `idxs`        | `jt.argmax(x, dim)` → `idxs`          |
| Comparison            | `torch.lt(a,b)`, etc.                  | `a < b`, `a >= b`, etc.               |
| Conditional selection | `torch.where(cond, x, y)`              | `jt.where(cond, x, y)`                |
| Transpose             | `x.t()`                                | `jt.transpose(x)`                     |
| Type cast             | `x.type(torch.float64)`                | `x.astype(jt.float64)`                |
| Matrix multiply       | `torch.mm(a, b)`                       | `a @ b` or `jt.matmul(a, b)`          |
| Shape                 | `x.size()[0]`                          | `jt.size(x)[0]`                       |
| Non-zero indices      | `torch.nonzero(x, as_tuple=False)`     | `jt.nonzero(x)`                       |

---

### 3. Learning Rate Scheduling (Warmup + Cosine Annealing)

**Files:** `models/functions/warmup.py` & `train_jittor.py`

* **Warmup**:

  * Replace PyTorch’s `_LRScheduler` base class with `jittor.optim.LRScheduler`.

* **Cosine Annealing**:

  * Jittor provides `CosineAnnealingLR`, but it lacks some features. A custom implementation in `warmup.py` might look like:

    ```python
    class CosineAnnealingLR(LRScheduler):
        def __init__(self, optimizer, T_max, eta_min=0):
            super().__init__(optimizer)
            self.T_max = T_max
            self.eta_min = eta_min
            # ... initialize
        def step(self):
            # compute and set new learning rate
    ```

---

### 4. Model & Module API Mappings

| PyTorch Module          | Jittor Module                             | Notes                                      |
| ----------------------- | ----------------------------------------- | ------------------------------------------ |
| `import torch.nn as nn` | `import jittor.nn as nn`                  | Direct swap                                |
| `forward(self, ...)`    | `execute(self, ...)`                      | Jittor’s default forward interface         |
| `nn.Conv2d(...)`        | `nn.Conv(...)` or `nn.Conv2d(...)`        | Both are identical in Jittor               |
| `nn.ReLU(inplace=True)` | `nn.ReLU()` or `nn.Relu()`                | In-place is default; `inplace` arg removed |
| `nn.Linear(...)`        | `nn.Linear(...)`                          | Identical                                  |
| `nn.BatchNorm2d(...)`   | `nn.BatchNorm2d(...)` or `nn.BatchNorm()` | Batch normalization                        |
| Normal init             | `nn.init.normal_(...)`                    | Use `nn.init.gauss_(...)`                  |

**Other mappings:**

* **NMS**:

  ```python
  # PyTorch:
  idxs = torchvision.ops.nms(boxes, scores, iou_thresh)

  # Jittor (dets = [x1,y1,x2,y2,score]):
  dets = jt.concat([boxes, scores.unsqueeze(1)], dim=1)
  idxs = jt.nms(dets, iou_thresh)
  ```

* **Pretrained weights**: Implement `load_pretrained_resnet_weights(path)` instead of `model_zoo.load_url()`.

* **Sampler**:

  ```python
  # PyTorch:
  sampler = torch.utils.data.SubsetRandomSampler(list(range(len(dataset))))

  # Jittor:
  sampler = jt.dataset.SubsetRandomSampler((dataset, 0, len(dataset)-1))
  ```

* **DataLoader**: Remove `pin_memory`; rename `collate_fn` → `collate_batch`.

---

### 5. Training & Optimization Differences

* **Backpropagation** (any of these three):

  ```python
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ```

  ```python
  optimizer.zero_grad()
  optimizer.backward(loss)
  optimizer.step()
  ```

  ```python
  optimizer.step(loss)
  ```
* **Grad access**: `p.grad` → `p.opt_grad()`.
* **Gradient clipping**:

  ```python
  # PyTorch
  nn.utils.clip_grad_norm_(model.parameters(), 0.1)

  # Jittor
  optimizer.clip_grad_norm(0.1)
  ```
* **Model save/load**:

  ```python
  model.save(path)
  model.load(path)
  ```

---

> The above covers the core changes needed to migrate DMANet from PyTorch to Jittor. Refer to the official documentation for further details and advanced usage.
