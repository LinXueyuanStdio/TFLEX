"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/2/20
@description: 实验的基础能力 demo
"""
import random

from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema):
        super(MyExperiment, self).__init__(output)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        seed = 10
        max_steps = 10
        learning_rate = 0.0001
        self.metric_log_store.set_rng_seed(seed)
        self.metric_log_store.add_hyper(learning_rate, "learning_rate")
        self.metric_log_store.add_progress(max_steps)
        for step in range(max_steps):
            acc = random.randint(10, 100)
            self.metric_log_store.add_loss(acc, step, name="loss")
            self.metric_log_store.add_metric({"acc": acc}, step, "Test")
            if acc > 50:
                self.metric_log_store.add_best_metric({"acc": acc}, "Test")
        self.metric_log_store.finish()


output = OutputSchema("demo")
MyExperiment(output)
