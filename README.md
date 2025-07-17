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



## 实验结果对齐：