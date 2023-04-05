import torch
from typing import Callable


class FGM(object):
    """
    # 对抗训练就是在输入的层次增加扰动，根据扰动产生的样本，来做一次反向传播。
    # 初始化
    fgm = FGM(model)
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., filter_by: Callable[[str], bool] = lambda name: 'emb.' in name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and filter_by(name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, filter_by: Callable[[str], bool] = lambda name: 'emb.' in name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and filter_by(name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
