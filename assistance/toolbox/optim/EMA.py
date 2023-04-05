import torch


class EMA:
    """
    移动平均，保存历史的一份参数，在一定训练阶段后，拿历史的参数给目前学习的参数做一次平滑。

    初始化
    ```
    ema = EMA(model, 0.999)
    ema.register()
    ```

    训练过程中，更新完参数后，同步 update shadow weights
    ```
    def train():
        optimizer.step()
        ema.update()
    ```

    eval 前，apply shadow weights；eval 之后，恢复原来模型的参数
    ```
    def evaluate():
        ema.apply_shadow()
        # evaluate
        ema.restore()
    ```
    """
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMASchedule:
    """
    移动平均，保存历史的一份参数，在一定训练阶段后，拿历史的参数给目前学习的参数做一次平滑。

    初始化
    ```
    ema = EMA(model, 0.999)
    ema_schedule = EMASchedule(ema, 1000, 100)
    ```

    训练过程中，更新完参数后，同步 update shadow weights
    ```
    def train():
        optimizer.step()
        ema_schedule.step(i)
    ```

    eval 前，apply shadow weights；eval 之后，恢复原来模型的参数
    ```
    def evaluate():
        ema_schedule.apply_shadow()
        # evaluate
        ema_schedule.restore()
    ```
    """
    def __init__(self, ema: EMA, step_start_ema: int, update_ema_every: int):
        self.ema = ema
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every

    def step(self, step: int):
        if step % self.update_ema_every == 0:
            if step < self.step_start_ema:
                self.ema.register()
            else:
                self.ema.update()

    def apply_shadow(self):
        self.ema.apply_shadow()

    def restore(self):
        self.ema.restore()


class EMA2:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
