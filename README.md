# DMANet-Jittor

> **Jittor 版本的 Dual Memory Aggregation Network (DMANet)**  
本项目是 [AAAI 2023 论文《Dual Memory Aggregation Network for Event-based Object Detection》](https://ojs.aaai.org/index.php/AAAI/article/view/25346) 的 **Jittor 框架复现版本**。

📌 原论文代码（PyTorch 实现）：  
👉 [https://github.com/wds320/AAAI_Event_based_detection](https://github.com/wds320/AAAI_Event_based_detection)


## 环境构建：

```
python -m jittor_utils.install_cuda
cd /root/.cache/jittor/cutlass
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip -O cutlass.zip
unzip cutlass.zip
```


可能的报错：raise RuntimeError(f"MD5 mismatch between the server and the downloaded file {file_path}")
RuntimeError: MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip，
cutlass对应的清华网盘无法访问，导致程序只创建了一个空压缩包，可以将cutlass的下载链接更换为[https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip]


## 实验结果对齐：