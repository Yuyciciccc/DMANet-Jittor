## DMANet PyTorch → Jittor 转换指南

本指南聚焦于将 DMANet 的 PyTorch 实现快速迁移至 Jittor，仅涵盖关键步骤及示例代码。详细配置与 API 可参阅 [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)。

---

### 1. 全局 CUDA 配置

* **PyTorch**：需要在每个 Tensor 或 Module 上显式调用 `.to(device)`。
* **Jittor**：通过全局开关统一设置，移除所有显式 `.to()`：

  ```python
  # train_jittor.py 或 AbstractTrainer/__init__.py
  import jittor as jt
  from config import settings  # 假设包含 gpu_device

  if settings.gpu_device != "cpu":
      jt.flags.use_cuda = 1
      jt.set_cuda_device(int(settings.gpu_device))
  ```

---

### 2. 核心 Tensor/Var 操作映射

| 操作         | PyTorch API                         | Jittor 对应 API                         |
| ---------- | ----------------------------------- | ------------------------------------- |
| 从 NumPy 创建 | `torch.from_numpy(ndarray)`         | `jt.array(ndarray)`                   |
| 限幅         | `torch.clamp(x, min=, max=)`        | `jt.clamp(x, min_v=, max_v=)`         |
| 指数         | `torch.exp(x)`                      | `jt.exp(x)`                           |
| 堆叠         | `torch.stack(list)`                 | `jt.stack(list)`                      |
| 元素最小/最大    | `torch.min(a,b)` / `torch.max(a,b)` | `jt.minimum(a,b)` / `jt.maximum(a,b)` |
| 全 1 张量     | `torch.ones(shape)`                 | `jt.ones(shape)`                      |
| 对数         | `torch.log(x)`                      | `jt.log(x)`                           |
| 最大值/索引     | `torch.max(x,dim)` → `(vals, idxs)` | `jt.max(x,dim)` → `vals` （仅值）         |
| \`argmax\` | `torch.argmax(x,dim)` → `idxs`      | `jt.argmax(x,dim)` → `idxs`           |
| 比较运算       | `torch.lt(a,b)` 等                   | 直接用运算符：`a < b`, `a >= b` 等            |
| 条件筛选       | `torch.where(cond, x, y)`           | `jt.where(cond, x, y)`                |
| 转置         | `x.t()`                             | `jt.transpose(x)`                     |
| 类型转换       | `x.type(torch.float64)`             | `x.astype(jt.float64)`                |
| 矩阵乘        | `torch.mm(a,b)`                     | `a @ b` 或 `jt.matmul(a,b)`            |
| 形状         | `x.size()[0]`                       | `jt.size(x)[0]`                       |
| 非零位置       | `torch.nonzero(x, as_tuple=False)`  | `jt.nonzero(x)`                       |

---

### 3. 学习率调度（Warmup + 余弦退火）

* **文件**：`models/functions/warmup.py` & `train_jittor.py`
* **Warmup**：

  * 将 PyTorch 的 `_LRScheduler` 替换为 `jittor.optim.LRScheduler`。
* **余弦退火**：

  * 虽然 Jittor 提供 `CosineAnnealingLR`，但功能尚不完善。可在 `warmup.py` 中自定义实现：

    ```python
    class CosineAnnealingLR(LRScheduler):
        def __init__(...):
            # 手动计算余弦退火学习率
        def step(self):
            # 更新 lr
    ```

---

### 4. 模型与模块 API 映射

| PyTorch 模块              | Jittor 模块                                   | 说明                       |
| ----------------------- | ------------------------------------------- | ------------------------ |
| `import torch.nn as nn` | `import jittor.nn as nn`                    | 通用替换                     |
| `forward(self, ...)`    | `execute(self, ...)`                        | Jittor 默认前向接口            |
| `nn.Conv2d(...)`        | `nn.Conv(...)` 或 `nn.Conv2d(...)`           | 两者等效，参数一致                |
| `nn.ReLU(inplace=True)` | `nn.ReLU()` 或 `nn.Relu()`                   | 默认 `inplace=True`，不支持该参数 |
| `nn.Linear(...)`        | `nn.Linear(...)`                            | 一致                       |
| `nn.BatchNorm2d(...)`   | `nn.BatchNorm2d(...)` 或 `nn.BatchNorm(...)` | 批归一化层                    |
| 权重初始化（Normal）           | `nn.init.normal_()`                         | 替换为 `nn.init.gauss_()`   |

**其他常用映射**：

* **NMS**：

  ```python
  # PyTorch
  idxs = torchvision.ops.nms(boxes, scores, iou_thresh)
  # Jittor：输入 [x1,y1,x2,y2,score]
  dets = jt.concat([boxes, scores.unsqueeze(1)], dim=1)
  idxs = jt.nms(dets, iou_thresh)
  ```

* **预训练权重加载**：自定义 `load_pretrained_resnet_weights(path)` 替代 `model_zoo.load_url()`。

* **Sampler**：

  ```python
  # PyTorch
  sampler = torch.utils.data.SubsetRandomSampler(list(range(len(dataset))))
  # Jittor
  sampler = jt.dataset.SubsetRandomSampler((dataset, 0, len(dataset)-1))
  ```

* **DataLoader**：移除 `pin_memory`，将 `collate_fn` 改为 `collate_batch`。

---

### 5. 训练与优化差异

* **反向传播**：三种等效写法：

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
* **参数梯度访问**：`p.grad` → `p.opt_grad()`。
* **梯度裁剪**：

  ```python
  # PyTorch
  nn.utils.clip_grad_norm_(model.parameters(), 0.1)
  # Jittor
  optimizer.clip_grad_norm(0.1)
  ```
* **模型保存/加载**：

  ```python
  model.save(path)
  model.load(path)
  ```

---

> 以上即将 DMANet PyTorch 版本迁移至 Jittor 的核心要点，更多细节请参阅官方 API 文档。
