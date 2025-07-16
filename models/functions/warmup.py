# import torch
# from torch.optim.lr_scheduler import _LRScheduler


# class WarmUpLR(_LRScheduler):
#     def __init__(self, optimizer, total_iters, last_epoch=-1):
#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         return [base_lr * self.last_epoch /(self.total_iters+1e-8) for base_lr in self.base_lrs]

import jittor as jt
from jittor.optim import LRScheduler

class WarmUpLR(LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def main():
    # 创建一个简单模型
    model = nn.Linear(10, 2)

    # 创建优化器
    optimizer = SGD(model.parameters(), lr=0.1)

    # 创建 WarmUpLR 学习率调度器（例如：前 5 步进行 warmup）
    warmup_scheduler = WarmUpLR(optimizer, total_iters=5)

    # 模拟训练过程
    print("📈 Warm-up Learning Rate Progression:")
    for epoch in range(10):
        # 模拟 forward + backward
        x = jt.randn(8, 10)
        y = model(x)
        loss = y.sum()
        optimizer.step(loss)

        # 调度器 step 更新学习率
        warmup_scheduler.step()

        # 打印当前学习率
        current_lr = [group['lr'] for group in optimizer.param_groups]
        print(f"Epoch {epoch}: lr = {current_lr}")

if __name__ == "__main__":
    from jittor import nn
    from jittor.optim import SGD
    main()