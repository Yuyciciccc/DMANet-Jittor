## DMANet-Jittor

> **Dual Memory Aggregation Network (DMANet) 的 Jittor 实现**

本项目基于 [AAAI 2023 论文《Dual Memory Aggregation Network for Event-based Object Detection》](https://ojs.aaai.org/index.php/AAAI/article/view/25346)，使用 Jittor 框架复现了事件相机目标检测模型。

* **原论文 PyTorch 实现**：[https://github.com/wds320/AAAI\_Event\_based\_detection](https://github.com/wds320/AAAI_Event_based_detection)
* **Jittor 文档**：[https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
* **详细转换细节**：`details.md`

---

### 一、环境搭建

1. 系统：Ubuntu 22.04 + CUDA 11.8 + RTX 4090 (24GB)
2. 创建并激活 Conda 环境：

   ```bash
   conda create -n dmanet-jittor python=3.8 -y
   conda activate dmanet-jittor
   ```
3. 安装依赖：

   ```bash
   pip install jittor==1.3.9.14 
   pip install -r requirements.txt
   ```
4. 验证 Jittor 安装：

   ```bash
   python -m jittor.test.test_example
   ```
5. **常见安装问题**

   > 报错：
   > `RuntimeError: MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip`

   原因：清华网盘下载的 `cutlass.zip` 文件为空。可手动下载并替换：

   ```bash
   python -m jittor_utils.install_cuda
   cd /root/.cache/jittor/cutlass
   wget https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip -O cutlass.zip
   unzip cutlass.zip
   ```

---

### 二、数据准备

* 数据集：1 Mpx Auto-Detection Sub Dataset (总大小约 268GB)
* 本实验只使用了约 4.25GB 的子集。

1. 下载链接（百度网盘）：

   * 链接：[https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA](https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA)
   * 提取码：`c6j9`
2. 数据目录结构：

   ```text
   prophesee_dlut
   ├── train
   │   ├── trainfilelist00…trainfilelist14
   ├── val
   │   ├── valfilelist00
   │   └── valfilelist01
   └── test
       ├── testfilelist00
       ├── testfilelist01
       └── testfilelist02
   ```
3. 可视化示例：

   ```bash
   python tools/data_check_npz.py --records /root/autodl-tmp/train/trainfilelist00
   ```

   ![Data Visualization](https://github.com/Yuyciciccc/DMANet-Jittor/blob/main/records/debug_0_0.png)

---

### 三、训练与测试

在 `settings.yaml` 中配置：

* `dataset_path`：数据集根目录
* `save_dir`：模型与日志保存路径

1. **训练**：

   ```bash
   python train_jittor.py --settings_file=path/to/settings.yaml
   ```
2. **测试**：
    在文件中修改读取模型的路径
   ```bash
   python test_jittor.py 
   ```

---

### 四、实验结果

本次实验pytorch和jittor均采用相同的数据集（约4.25G）
具体的实验配置可以在settings.yaml中查看

所有日志保存在 `records/` 目录：

* `checkpoints/`：Loss 曲线（可通过 TensorBoard 查看）
```bash
tensorboard --logdir=/path/to/
```
* `train_log/`：数据加载、前向/后向耗时 实验记录
* `test_result/`：测试输出 
*  `settings.yaml`: 实验配置

#### 1. 运行性能比较

| Framework | 前向时间 / batch (s) | 后向时间 / batch (s) | 总时间 / batch (s) |
| --------- | ---------------- | ---------------- | --------------- |
| PyTorch   | 0.5167           | 0.4309           | 1.0116          |
| Jittor    | 0.5876           | 0.1228           | 0.7221          |

#### 2. 检测精度对比

* **PyTorch** 实验结果（180 张图像，共 1010 个标注）：

| Class         | Images | Labels | Precision | Recall  | mAP@0.5 | mAP@0.5:0.95 |
|---------------|--------|--------|-----------|---------|---------|--------------|
| all           | 180    | 1010   | 0.04180   | 0.05230 | 0.01772 | 0.00390      |
| pedestrian    | 180    | 155    | 0.00757   | 0.00645 | 0.00239 | 0.00033      |
| two wheeler   | 180    | 35     | 0.00000   | 0.00000 | 0.00086 | 0.00011      |
| car           | 180    | 799    | 0.15962   | 0.20275 | 0.06764 | 0.01517      |
| truck         | 180    | 21     | 0.00000   | 0.00000 | 0.00000 | 0.00000      |


* **Jittor** 实验结果：

| Class         | Images | Labels | Precision | Recall  | mAP@0.5 | mAP@0.5:0.95 |
|---------------|--------|--------|-----------|---------|---------|--------------|
| all           | 180    | 1010   | 0.04220   | 0.06101 | 0.01858 | 0.00369      |
| pedestrian    | 180    | 155    | 0.00641   | 0.00645 | 0.00201 | 0.00034      |
| two wheeler   | 180    | 35     | 0.02737   | 0.02857 | 0.00297 | 0.00030      |
| car           | 180    | 799    | 0.13501   | 0.20901 | 0.06928 | 0.01410      |
| truck         | 180    | 21     | 0.00000   | 0.00000 | 0.00008 | 0.00001      |

---

### 五、参考

* [AAAI 2023 论文](https://ojs.aaai.org/index.php/AAAI/article/view/25346)
* [原始 PyTorch 代码](https://github.com/wds320/AAAI_Event_based_detection)
* [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
