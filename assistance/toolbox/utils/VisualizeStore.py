"""
@date: 2022/2/19
@description: 可视化
run the command below to open tensorbard
```shell
tensorboard --logdir .
```
"""


def get_writer(log_dir: str, comments=""):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir, comments)


def add_scalar(writer, name: str, value, step_num: int):
    writer.add_scalar(name, value, step_num)


def add_result(writer, result, step_num: int):
    left2right = result["left2right"]
    right2left = result["right2left"]
    using_time = result["time"]
    sorted(left2right)
    sorted(right2left)
    for i in left2right:
        add_scalar(writer, i, left2right[i], step_num)
    for i in right2left:
        add_scalar(writer, i, right2left[i], step_num)
    add_scalar(writer, "using time (s)", using_time, step_num)


class VisualizeStoreSchema:
    def __init__(self, log_dir: str, comments=""):
        self.writer = get_writer(log_dir, comments)
        print()
        print("Tensorboard is activated on dir " + log_dir)
        print("You can open tensorboard with:")
        print("     tensorboard --logdir " + log_dir + " --host=your_ip --port=6006")
        print()

    def add_scalar(self, name: str, value, step_num: int):
        add_scalar(self.writer, name, value, step_num)

    def add_model(self, model):
        self.writer.add_graph(model)

    def add_embedding(self, embedding, labels=None, step_num: int = 0):
        # fix:
        # module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem'
        # see:
        # https://github.com/pytorch/pytorch/issues/30966
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        self.writer.add_embedding(embedding, metadata=labels, global_step=step_num)

    def add_result(self, result, step_num: int):
        add_result(self.writer, result, step_num)

    def add_link_prediction_result(self, result, step_num: int, scope: str):
        for key in result:
            for i in result[key]:
                self.add_scalar(f"{scope}_{key}_{i}", result[key][i], step_num)
