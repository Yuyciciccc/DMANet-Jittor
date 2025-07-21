# DMANet-Jittor

> **Jittor 版本的 Dual Memory Aggregation Network (DMANet)**  
本项目是 [AAAI 2023 论文《Dual Memory Aggregation Network for Event-based Object Detection》](https://ojs.aaai.org/index.php/AAAI/article/view/25346) 的 **Jittor 框架复现版本**。

📌 原论文代码（PyTorch 实现）：  
👉 [https://github.com/wds320/AAAI_Event_based_detection](https://github.com/wds320/AAAI_Event_based_detection)

参考文档如下：
[https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)

## 环境构建：
该实验基于Ubuntu系统22.04 cuda11.8 RTX4090(24GB)*1

```
conda create -n dmanet-jittor python=3.8
conda activate dmanet-jittor
pip install jittor==1.3.9.14
pip install torch
pip install -r requirements.txt
```

你可以使用以下代码检查jittor是否安装完成
```
python -m jittor.test.test_example
```

可能的报错：raise RuntimeError(f"MD5 mismatch between the server and the downloaded file {file_path}")
RuntimeError: MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip，

原因：cutlass对应的清华网盘无法访问，导致程序只创建了一个空压缩包，可以将cutlass的下载链接更换为[https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip]
```
python -m jittor_utils.install_cuda
cd /root/.cache/jittor/cutlass
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip -O cutlass.zip
unzip cutlass.zip
```

## 数据准备：
- 1 Mpx Auto-Detection Sub Dataset 

- Download 1 Mpx Auto-Detection Sub Dataset. (Total 268GB)

Links: [https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA](https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA)

Password: c6j9 

在本次任务中，由于算力的限制，只使用了极少量数据进行实验
- Dataset structure
```
prophesee_dlut   
├── test
│   ├── testfilelist00
│   ├── testfilelist01
│   └── testfilelist02
├── train
│   ├── trainfilelist00
│   ├── trainfilelist01
│   ├── trainfilelist02
│   ├── trainfilelist03
│   ├── trainfilelist04
│   ├── trainfilelist05
│   ├── trainfilelist06
│   ├── trainfilelist07
│   ├── trainfilelist08
│   ├── trainfilelist09
│   ├── trainfilelist10
│   ├── trainfilelist11
│   ├── trainfilelist12
│   ├── trainfilelist13
│   └── trainfilelist14
└── val
    ├── valfilelist00
    └── valfilelist01
```
## 数据集可视化

```
python tools/data_check_npz.py --records /root/autodl-tmp/train/trainfilelist00
```
![图片](https://github.com/Yuyciciccc/DMANet-Jittor/blob/main/records/debug_0_0.png)
可视化结果示例如下：

## 训练 & 测试：
Change settings.yaml, including *dataset_path* and *save_dir*.  

- 1. Training
```
python train-jittor.py --settings_file=$YOUR_YAML_PATH
```
- 2. Testing
```
python test.py --weight=$YOUR_MODEL_PATH
```