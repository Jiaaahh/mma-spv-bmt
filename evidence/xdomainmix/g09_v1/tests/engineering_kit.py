"""G09-A 工程测试套件（**仅 CPU 桩件**，不是 GPU 资格证据）。

- `TinyNet`：小卷积网络（含 BN 与 `fc` 分类头），用于快速验证合同/优化器/序列化/拒绝逻辑。
- `StubAlgo`：XDM 桩算法，仅复刻**接口与接线**（hparams 引用、class/domain 双优化器、
  mix 计数与冻结作者浮点阈值循环、`loss_task = 0.5*loss1 + 0.5*loss2` 的字段形态），
  **不是**作者的 ResNet50 混合实现，不能用于任何数值结论。

用这两个桩件跑出的任何数字都只证明代码路径，不能被引用为真实训练/资格结果。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

STUB_NOTICE = 'CPU_TEST_STUB_ONLY__NOT_GPU_QUALIFICATION'


class TinyNet(nn.Module):
    """(B,3,224,224) -> 180 类；`body` 含 BN 以覆盖冻结 BN 与可训仿射参数两条路径。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(4, 180)

    def forward(self, x):
        return self.fc(self.body(x))


class TinyDomainHead(nn.Module):
    def __init__(self, features=4, domains=5):
        super().__init__()
        self.classifier = nn.Linear(features, domains)

    def forward(self, x):
        return self.classifier(x)


class StubMixupModule:
    """冻结作者阈值循环的最小复刻（只保留阈值/计数语义）。"""

    def __init__(self):
        self.call_num = 0
        self.threshold = 0.9
        self.threshold_lower_bound = 0.5
        self.value_to_change = 0.1
        self.step_to_change = 100

    def note_mix_call(self):
        """在**本步混合之前**调用，与作者 `before_mix = max(0, step-3001)` 的算术一致。"""
        before = self.call_num
        if before != 0 and before % self.step_to_change == 0:
            nxt = self.threshold - self.value_to_change
            if self.threshold == self.threshold_lower_bound:
                self.threshold = 0.9
            elif nxt < self.threshold_lower_bound:
                self.threshold = self.threshold_lower_bound
            else:
                self.threshold = nxt
        self.call_num += 1


class _StubNetwork(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.network = TinyNet()
        self.domain_classifier = TinyDomainHead()
        self.mixup_module = StubMixupModule()


class StubAlgo(nn.Module):
    """XDM 桩：接口与 v1 一致，科学内容为占位实现。"""

    def __init__(self, hparams, device='cpu'):
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams['batch_size']
        self.network = _StubNetwork(hparams)
        self.optimizer = torch.optim.Adam(self.network.network.parameters(), lr=hparams['lr'],
                                          weight_decay=hparams['weight_decay'])
        self.domain_optimizer = torch.optim.Adam(self.network.domain_classifier.parameters(),
                                                 lr=hparams['lr'], weight_decay=hparams['weight_decay'])
        self.class_gradient = None
        self.domain_gradient = None
        self.call_num = 0
        self.to(device)

    def predict(self, x):
        return self.network.network(x)

    def _mixed(self, x):
        index = torch.arange(x.shape[0] - 1, -1, -1)
        return 0.5 * x + 0.5 * x[index]

    def update(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        if self.call_num < self.hparams['warmup_step']:
            loss_class = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad(); loss_class.backward(); self.optimizer.step()
            self.call_num += 1
            return {'loss_task': float(loss_class.detach().cpu())}
        mix = self.network.mixup_module
        mix.note_mix_call()
        loss1 = F.cross_entropy(self.predict(all_x), all_y)
        loss2 = F.cross_entropy(self.predict(self._mixed(all_x)), all_y)
        loss_task = 0.5 * loss1 + 0.5 * loss2
        self.optimizer.zero_grad(); loss_task.backward(); self.optimizer.step()
        features = self.network.network.body(all_x).detach()
        domain_labels = torch.arange(5, device=all_x.device).repeat_interleave(self.batch_size)
        loss_domain = F.cross_entropy(self.network.domain_classifier(features), domain_labels)
        self.domain_optimizer.zero_grad(); loss_domain.backward(); self.domain_optimizer.step()
        self.call_num += 1
        return {'loss_task': float(loss_task.detach().cpu()), 'loss_domain': float(loss_domain.detach().cpu())}
