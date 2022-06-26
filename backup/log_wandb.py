"""
@date: 2022/3/9
@description: null
"""
import random

import wandb

wandb.init(project="FLEX3", entity="bupt817")
wandb.run.log_code("toolbox")
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}
all_tasks = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']

for step in range(100):
    loss = 100 - step + random.randint(0, 5)
    if step % 10 == 0:
        my_custom_data = []
        for task in all_tasks:
            d = {
                'MRR': random.randint(200, 300) / 500,
                'hits@1': random.randint(000, 200) / 500,
                'hits@3': random.randint(200, 300) / 500,
                'hits@10': random.randint(300, 500) / 500,
                'num_queries': random.randint(30000, 50000),
            }
            my_custom_data.append([v for k, v in d.items()])
        wandb.log({"valid_custom_data_table": wandb.Table(data=my_custom_data, columns = ["MRR", "hits@1", "hits@3", "hits@10", "num_queries"], rows=all_tasks)}, step=step)

        my_custom_data = []
        for task in all_tasks:
            d = {
                'MRR': random.randint(200, 300) / 500,
                'hits@1': random.randint(000, 200) / 500,
                'hits@3': random.randint(200, 300) / 500,
                'hits@10': random.randint(300, 500) / 500,
                'num_queries': random.randint(30000, 50000),
            }
            my_custom_data.append([v for k, v in d.items()])
        wandb.log({"test_custom_data_table": wandb.Table(data=my_custom_data, columns = ["MRR", "hits@1", "hits@3", "hits@10", "num_queries"], rows=all_tasks)}, step=step)
    wandb.log({"loss": loss}, step=step)
