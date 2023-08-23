"""
@date: 2021/10/26
@description: null
"""
import click

from ComplexTemporalQueryData import ComplexTemporalQueryDatasetCachePath, TemporalComplexQueryData
from run_migration_QE_to_TQE import FB15k_237_TFLEX, FB15k_TFLEX, NELL_TFLEX
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.RandomSeeds import set_seeds
from train_TCQE_TFLEX import *


class TFLEX_static(TFLEX):
    def __init__(self, nentity, nrelation, ntimestamp, hidden_dim, gamma,
                 test_batch_size=1,
                 center_reg=None, drop: float = 0.):
        super(TFLEX_static, self).__init__(nentity, nrelation, ntimestamp, hidden_dim, gamma, test_batch_size, center_reg, drop)
        self.timestamp_feature_embedding.requires_grad_(False)

    def init(self):
        super(TFLEX_static, self).init()
        nn.init._no_grad_zero_(self.timestamp_feature_embedding.weight)


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="ICEWS14", help="Which dataset to use: ICEWS14, ICEWS05_15, GDELT.")
@click.option("--name", type=str, default="TFLEX_base", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=200001, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10000, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=10000, help="Number of steps.")
@click.option("--batch_size", type=int, default=512, help="Batch size.")
@click.option("--test_batch_size", type=int, default=8, help="Test batch size.")
@click.option('--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0,
              help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.0001, help="Learning rate.")
@click.option('--cpu_num', type=int, default=1, help="used to speed up torch.dataloader")
@click.option('--hidden_dim', type=int, default=800, help="embedding dimension")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option('--gamma', type=float, default=15.0, help="margin in the loss")
@click.option('--center_reg', type=float, default=0.02,
              help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
@click.option('--train_tasks', type=str, default="Pe,Pe2,Pe3,e2i,e3i,"
              + "e2i_N,e3i_N,Pe_e2i_Pe_NPe,e2i_PeN,e2i_NPe", help='the tasks for training')
@click.option('--train_all', type=bool, default=False,
              help='if training all, it will use all tasks in data.train_queries_answers')
@click.option('--eval_tasks', type=str, default="Pe,Pt,Pe2,Pe3", help='the tasks for evaluation')
@click.option('--eval_all', type=bool, default=False,
              help='if evaluating all, it will use all tasks in data.test_queries_answers')
@click.option("--ts", type=int, default=0, help="0 for one fake timestamp, n>0 for up to n fake timestamps.")
def main(data_home, dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, negative_sample_size,
         train_device, test_device,
         resume, resume_by_score,
         lr, cpu_num,
         hidden_dim, input_dropout, gamma, center_reg, train_tasks, train_all, eval_tasks, eval_all,
         ts,
         ):
    set_seeds(0)
    output = OutputSchema(dataset + "-" + name)
    from pathlib import Path
    suffix = "simple" if ts == 0 else f"time_{ts}"
    data_home = data_home if ts == 0 else Path(data_home) / suffix
    if dataset == "FB15k-237":
        dataset = FB15k_237_TFLEX(data_home)
    elif dataset == "FB15k":
        dataset = FB15k_TFLEX(data_home)
    elif dataset == "NELL":
        dataset = NELL_TFLEX(data_home)
    cache = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
    data = TemporalComplexQueryData(dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache([
        "meta",
    ])

    data.timestamp_count += 1
    entity_count = data.entity_count
    relation_count = data.relation_count // 2
    timestamp_count = data.timestamp_count
    max_relation_id = relation_count
    model = TFLEX(
        nentity=entity_count,
        nrelation=relation_count + max_relation_id,  # with reverse relations
        ntimestamp=timestamp_count,
        hidden_dim=hidden_dim,
        gamma=gamma,
        center_reg=center_reg,
        test_batch_size=test_batch_size,
        drop=input_dropout,
    )
    MyExperiment(
        output, data, model,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, negative_sample_size,
        train_device, test_device,
        resume, resume_by_score,
        lr, cpu_num,
        hidden_dim, input_dropout, gamma, center_reg, train_tasks, train_all, eval_tasks, eval_all
    )


if __name__ == '__main__':
    main()
