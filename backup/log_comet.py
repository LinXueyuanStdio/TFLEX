"""
@date: 2022/3/9
@description: null
"""
# Import comet_ml at the top of your file
import random

from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="P8uZ9naSu0vm4UVWHmDna3ltY",
    project_name="flex",
    workspace="linxueyuanstdio",
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 0.5,
    "epochs": 5,
    "steps": 100,
    "batch_size": 50,
}
experiment.log_parameters(hyper_params)

# Or report single hyperparameters:
hidden_layer_size = 50
experiment.log_parameter("hidden_layer_size", hidden_layer_size)
all_tasks = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']
# Long any time-series metrics:
for epoch in range(1, hyper_params["epochs"] + 1):
    for step in range(1, hyper_params["steps"] + 1):
        loss = hyper_params["steps"] - step + random.randint(0, 5)
        if step % 10 == 0:
            for task in all_tasks:
                experiment.log_metrics({
                    'MRR': random.randint(200, 300) / 500,
                    'hits@1': random.randint(000, 200) / 500,
                    'hits@3': random.randint(200, 300) / 500,
                    'hits@10': random.randint(300, 500) / 500,
                    'num_queries': random.randint(30000, 50000),
                }, prefix="valid", epoch=epoch, step=step)
        if step % 20 == 0:
            experiment.log_metrics({
                'MRR': random.randint(200, 300) / 500,
                'hits@1': random.randint(000, 200) / 500,
                'hits@3': random.randint(200, 300) / 500,
                'hits@10': random.randint(300, 500) / 500,
                'num_queries': random.randint(30000, 50000),
            }, prefix="test", epoch=epoch, step=step)
        experiment.log_metric("loss", loss, epoch=epoch, step=step)
