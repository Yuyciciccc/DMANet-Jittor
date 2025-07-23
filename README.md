## DMANet-Jittor

> **Dual Memory Aggregation Network (DMANet) Implementation in Jittor**

This project reproduces the event-based object detection model from the AAAI 2023 paper [*Dual Memory Aggregation Network for Event-based Object Detection*](https://ojs.aaai.org/index.php/AAAI/article/view/25346) using the Jittor framework.

* **Original PyTorch Implementation**: [https://github.com/wds320/AAAI\_Event\_based\_detection](https://github.com/wds320/AAAI_Event_based_detection)
* **Jittor Documentation**: [https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
* **Detailed Conversion Notes**: `details_en.md`

---

### 1. Environment Setup

1. **System**: Ubuntu 22.04 + CUDA 11.8 + RTX 4090 (24 GB)
2. **Create & Activate Conda Env**:

   ```bash
   conda create -n dmanet-jittor python=3.8 -y
   conda activate dmanet-jittor
   ```
3. **Install Dependencies**:

   ```bash
   pip install jittor==1.3.9.14
   pip install -r requirements.txt
   ```
4. **Verify Jittor**:

   ```bash
   python -m jittor.test.test_example
   ```
5. **Common Installation Issue**

   > **Error:**
   >
   > ```
   > RuntimeError: MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip
   > ```

   **Cause:** The `cutlass.zip` downloaded from the mirror is empty. Manually download and replace:

   ```bash
   python -m jittor_utils.install_cuda
   cd /root/.cache/jittor/cutlass
   wget https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip -O cutlass.zip
   unzip cutlass.zip
   ```

---

### 2. Data Preparation

* **Dataset**: 1 Mpx Auto-Detection Sub Dataset (\~268 GB total)
* **This Experiment**: uses a \~4.25 GB subset

1. **Download (Baidu Netdisk)**:

   * Link: [https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA](https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA)
   * Extraction Code: `c6j9`
2. **Directory Structure**:

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
3. **Visual Inspection**:

   ```bash
   python tools/data_check_npz.py --records /root/autodl-tmp/train/trainfilelist00
   ```

   ![Data Visualization](https://github.com/Yuyciciccc/DMANet-Jittor/blob/main/records/debug_0_0.png)

---

### 3. Training & Testing

Configure the following in `settings.yaml`:

* `dataset_path`: root of the dataset
* `save_dir`: directory for logs & checkpoints

1. **Train**:

   ```bash
   python train_jittor.py --settings_file=path/to/settings.yaml
   ```
2. **Test**:

   * Edit `test_jittor.py` to point to the correct checkpoint path

   ```bash
   python test_jittor.py
   ```

---

### 4. Experimental Results

Both PyTorch and Jittor experiments used the same \~4.25 GB subset. Details in `settings.yaml`.
Logs and outputs are saved under `records/`:

* `checkpoints/`: TensorBoard logs for loss curves

  ```bash
  tensorboard --logdir=/path/to/records/checkpoints
  ```
* `train_log/`: data loading and forward/backward timing
* `test_result/`: inference outputs

#### 4.1 Runtime Performance Comparison

| Framework | Forward Time (s/batch) | Backward Time (s/batch) | Total Time (s/batch) |
| --------- | ---------------------- | ----------------------- | -------------------- |
| PyTorch   | 0.5167                 | 0.4309                  | 1.0116               |
| Jittor    | 0.5876                 | 0.1228                  | 0.7221               |

#### 4.2 Detection Accuracy Comparison

**PyTorch Results** (180 images, 1010 annotations):

| Class       | Images | Labels | Precision | Recall  | mAP\@0.5 | mAP\@0.5:0.95 |
| ----------- | ------ | ------ | --------- | ------- | -------- | ------------- |
| all         | 180    | 1010   | 0.04180   | 0.05230 | 0.01772  | 0.00390       |
| pedestrian  | 180    | 155    | 0.00757   | 0.00645 | 0.00239  | 0.00033       |
| two wheeler | 180    | 35     | 0.00000   | 0.00000 | 0.00086  | 0.00011       |
| car         | 180    | 799    | 0.15962   | 0.20275 | 0.06764  | 0.01517       |
| truck       | 180    | 21     | 0.00000   | 0.00000 | 0.00000  | 0.00000       |

**Jittor Results**:

| Class       | Images | Labels | Precision | Recall  | mAP\@0.5 | mAP\@0.5:0.95 |
| ----------- | ------ | ------ | --------- | ------- | -------- | ------------- |
| all         | 180    | 1010   | 0.04220   | 0.06101 | 0.01858  | 0.00369       |
| pedestrian  | 180    | 155    | 0.00641   | 0.00645 | 0.00201  | 0.00034       |
| two wheeler | 180    | 35     | 0.02737   | 0.02857 | 0.00297  | 0.00030       |
| car         | 180    | 799    | 0.13501   | 0.20901 | 0.06928  | 0.01410       |
| truck       | 180    | 21     | 0.00000   | 0.00000 | 0.00008  | 0.00001       |

---

### 5. References

* AAAI 2023 Paper: *Dual Memory Aggregation Network for Event-based Object Detection*
* Original PyTorch Code: [https://github.com/wds320/AAAI\_Event\_based\_detection](https://github.com/wds320/AAAI_Event_based_detection)
* Jittor Docs: [https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
