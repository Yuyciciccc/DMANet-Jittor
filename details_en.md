Detailed guide on converting the PyTorch version of DMANet to Jittor, covering only the key parts.
Reference: [https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)

## CUDA setup

* **PyTorch** uses explicit calls: every Tensor or Module must call `.to(device)`.
* **Jittor** uses a global switch; set it once at the start of your training script.
  To convert, remove all explicit `.to(device)` calls and add:

  ```python
  # in train_jittor.py or AbstractTrainer/__init__.py
  if settings.gpu_device != "cpu":
      jt.flags.use_cuda = 1
      jt.set_cuda_device(int(settings.gpu_device))
  ```

## Tensor → Var operation mappings

* **NumPy → tensor**
  `torch.from_numpy()` → `jt.array()`
* **Clamp values**
  `torch.clamp(tensor, min=…, max=…)` → `jt.clamp(var, min_v=…, max_v=…)`
* **Exponential**
  `torch.exp()` → `jt.exp()`
* **Stack tensors**
  `torch.stack()` → `jt.stack()`
* **Element‑wise min/max**
  `torch.min(a, b)` → `jt.minimum(a, b)`
  `torch.max(a, b)` → `jt.maximum(a, b)`
* **All‑ones tensor**
  `torch.ones()` → `jt.ones()`
* **Logarithm**
  `torch.log()` → `jt.log()`
* **Max value & index**
  PyTorch’s `torch.max()` returns `(max_val, idx)`; Jittor’s `jt.max()` returns only `max_val`.
  PyTorch’s `torch.argmax()` returns `idx`; Jittor’s `jt.argmax()` returns `(idx, max_val)`.
* **Comparison functions**
  No direct `torch.lt`/`torch.ne`, etc.—use operators:

  ```python
  # PyTorch
  targets[torch.lt(IoU_max, 0.4), :] = 0
  positive_indices = torch.ge(IoU_max, 0.5)
  # Jittor
  targets[IoU_max < 0.4, :] = 0
  positive_indices = IoU_max >= 0.5
  ```
* **Conditional selection**
  `torch.where(cond, x, y)` → `jt.where(cond, x, y)`
* **Transpose**
  `tensor.t()` → `jt.transpose(tensor)`
* **Type casting**
  `tensor.type(torch.float64)` → `var.astype(jt.float64)` (and similarly for other dtypes)
* **Matrix multiplication**
  `torch.mm(a, b)` → `jt.matmul(a, b)` or `a @ b` (Jittor has no `mm()`)
* **Size**
  `tensor.size()[0]` → `jt.size(var)[0]`
* **Non‑zero indices**
  `torch.nonzero(as_tuple=False)` → `jt.nonzero()` (no `as_tuple` param)

## Learning‑rate scheduling

Main files: `models/functions/warmup.py`, `train_jittor.py`
Two-stage LR strategy:

1. **Warm‑up**: replace the PyTorch `_LRScheduler` base class with `jittor.optim.LRScheduler`.
2. **Cosine annealing**: Jittor’s built-in `CosineAnnealingLR` is incomplete, so a custom version is implemented in `models/functions/warmup.py`.

## Model conversion

* **Import**
  `import torch.nn as nn` → `import jittor.nn as nn`
* **Forward method**
  `def forward(self, …):` → `def execute(self, …):`
* **Convolution**
  `nn.Conv2d(ch, ch, k, padding=(k-1)//2)` → `nn.Conv(ch, ch, k, padding=(k-1)//2)` (or `nn.Conv2`)
  (In Jittor, `Conv` and `Conv2d` are identical.)
* **ReLU**
  `nn.ReLU(inplace=True)` → `nn.Relu()` or `nn.ReLU()` (Jittor defaults to inplace; no such arg.)
* **Linear**
  `nn.Linear()` → `nn.Linear()`
* **BatchNorm**
  `nn.BatchNorm2d()` → `nn.BatchNorm()`, `nn.BatchNorm1d()`, `nn.BatchNorm2d()`, or `nn.BatchNorm3d()`
* **Gaussian weight init**
  `nn.init.normal_(m.weight, mean, std)` → `nn.init.gauss_(m.weight, mean, std)`
* **NMS**
  `torchvision.ops.nms(boxes, scores, iou_thresh)` → `jt.nms(dets, iou_thresh)`
  where `dets` is `[N,5]` combining `[boxes, scores]`:

  ```python
  anchors_nms_idx = jt.nms(
      jt.concat([anchorBoxes, scores.unsqueeze(1)], dim=1),
      self.iou_threshold
  )
  ```
* **Loading pretrained ResNet weights**
  PyTorch: `model_zoo.load_url(...)`
  Jittor: custom `load_pretrained_resnet_weights(...)`

## Dataset & DataLoader

* **SubsetRandomSampler**
  PyTorch:

  ```python
  torch.utils.data.sampler.SubsetRandomSampler(list(range(len(train_dataset))))
  ```

  Jittor:

  ```python
  jt.dataset.SubsetRandomSampler(train_dataset, (0, len(train_dataset) - 1))
  ```
* **DataLoader changes**

  * Remove `pin_memory` argument.
  * Rename `collate_fn` → `collate_batch`.

## Backpropagation examples

All three work in Jittor:

```python
# 1
optimizer.zero_grad()
loss.backward()
optimizer.step()
# 2
optimizer.zero_grad()
optimizer.backward(loss)
optimizer.step()
# 3
optimizer.step(loss)
```

## Gradient access & clipping

* **Access gradients**: PyTorch `p.grad` → Jittor `p.opt_grad()`
* **Gradient clipping**:
  PyTorch: `nn.utils.clip_grad_norm_(model.parameters(), 0.1)`
  Jittor: `optimizer.clip_grad_norm(0.1)`

## Saving & loading models

```python
model.save()
model.load()
```
