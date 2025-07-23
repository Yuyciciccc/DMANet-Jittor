import jittor as jt
from jittor.optim import LRScheduler
from jittor.optim import Adam
import math
from typing import Optional

class WarmUpLR(LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class CosineAnnealingLR:
    """
    自定义 Cosine Annealing 学习率调度器，支持多周期平滑衔接

    参数:
        optimizer: 需要调度的优化器，拥有 param_groups 属性
        T_max: 一个周期内的步数（通常为 epoch 数）
        eta_min: 最低学习率
        last_epoch: 上一次 epoch 索引，默认为 -1（初始化前）
    """
    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):

        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._step_count = 0  # 记录 step 调用次数
        # 记录每个参数组的初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # 初始化：如果用户传入 last_epoch >= 0，则提前设置学习率
        if self.last_epoch >= 0:
            # 将 step_count 设为 1 以使 get_lr 中的第一次退火逻辑生效
            self._step_count = 1
            # 根据封闭公式直接设置到指定 epoch 的 lr
            for idx, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self._compute_closed_form(self.base_lrs[idx], self.last_epoch)

    def _compute_closed_form(self, base_lr: float, epoch: int) -> float:

        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2

    def get_lr(self, base_lr: float, prev_lr: float) -> float:
        """
        根据上一步 lr (prev_lr) 和当前 epoch (last_epoch) 计算新 lr
        """
        # 第一次 step(): 直接返回 optimizer 当前设置的 lr (初始 lr)
        if self.last_epoch == 0:
            return prev_lr

        # 第一次真正退火
        if self._step_count == 1 and self.last_epoch > 0:
            return self._compute_closed_form(base_lr, self.last_epoch)

        # 周期边界平滑衔接
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            delta = (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
            return prev_lr + delta

        # 其他步用增量比例
        num = 1 + math.cos(math.pi * self.last_epoch / self.T_max)
        den = 1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)
        return (num / den) * (prev_lr - self.eta_min) + self.eta_min

    def step(self, epoch: Optional[int] = None) -> None:
        """
        更新学习率；如果传入 epoch，则重置 last_epoch
        """
        self._step_count += 1
        # 支持手动设置 epoch
        if epoch is not None:
            if epoch < 0:
                raise ValueError(f"Epoch must be non-negative, got {epoch}")
            self.last_epoch = epoch
        else:
            self.last_epoch += 1

        # 更新每个参数组的学习率
        for idx, group in enumerate(self.optimizer.param_groups):
            prev = group['lr']
            base_lr = self.base_lrs[idx]
            group['lr'] = self.get_lr(base_lr, prev)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(T_max={self.T_max}, eta_min={self.eta_min}, "
            f"last_epoch={self.last_epoch}, step_count={self._step_count})"
        )

def test_lr_schedulers(n_batches=76, warm_epochs=1, total_epochs=10, init_lr=0.001):
    """
    Simulate WarmUpLR and CosineAnnealingLR schedules over training.

    Args:
        n_batches (int): number of batches per epoch
        warm_epochs (int): number of epochs to warm up
        total_epochs (int): total number of epochs
        init_lr (float): initial learning rate
    """
    # Dummy parameter
    param = jt.zeros([1])
    optimizer = Adam([param], lr=init_lr)

    # Initialize schedulers
    warmup_steps = n_batches * warm_epochs
    warmup = WarmUpLR(optimizer, warmup_steps)
    cosine_steps = n_batches * (total_epochs - warm_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=init_lr * 0.1
    )

    # Record LR
    lrs = []
    total_steps = n_batches * total_epochs
    for step in range(total_steps):
        if step < warmup_steps:
            warmup.step()
        else:
            cosine.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Print schedule
    for i, lr in enumerate(lrs, 1):
        print(f"Step {i:4d}: LR = {lr:.8f}")


if __name__ == '__main__':
    # Example: 76 batches/epoch, 1 warm-up epoch, 10 total epochs, init_lr=1e-3
    test_lr_schedulers(n_batches=76, warm_epochs=1, total_epochs=10, init_lr=0.001)


