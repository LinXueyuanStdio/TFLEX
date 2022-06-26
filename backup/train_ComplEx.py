"""
@date: 2022/4/13
@description: null
"""
from collections import defaultdict
from typing import List

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from toolbox.data.DataSchema import RelationalTripletData, RelationalTripletDatasetCachePath
from toolbox.data.DatasetSchema import get_dataset
from toolbox.data.LinkPredictDataset import LinkPredictDataset, LinkPredictTypeConstraintDataset
from toolbox.data.TripleDataset import TripleDataset
from toolbox.data.functional import with_inverse_relations, build_map_hr_t
from toolbox.evaluate.Evaluate import get_score
from toolbox.evaluate.LinkPredict import batch_link_predict2, as_result_dict2, batch_link_predict_type_constraint
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.nn.Regularizer import N3
from toolbox.optim.lr_scheduler import get_scheduler
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds


class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim,
                 reg_weight=0.001,
                 ):
        super(ComplEx, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.Ea = nn.Embedding(num_entities, entity_dim)
        self.Eb = nn.Embedding(num_entities, entity_dim)
        self.Ra = nn.Embedding(num_relations, relation_dim)
        self.Rb = nn.Embedding(num_relations, relation_dim)

        self.regularizer = N3(reg_weight)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def init(self):
        nn.init.kaiming_uniform_(self.Ea.weight.data)
        nn.init.kaiming_uniform_(self.Eb.weight.data)
        nn.init.kaiming_uniform_(self.Ra.weight.data)
        nn.init.kaiming_uniform_(self.Rb.weight.data)

    def forward(self, h_idx, r_idx, t_idx):
        h_idx = h_idx.view(-1)
        r_idx = r_idx.view(-1)
        t_idx = t_idx.view(-1)
        ha, hb = self.Ea(h_idx), self.Eb(h_idx)
        ra, rb = self.Ra(r_idx), self.Rb(r_idx)
        ta, tb = self.Ea(t_idx), self.Eb(t_idx)
        Ea, Eb = self.Ea.weight, self.Eb.weight

        score_tail = self.score_tail(ha, hb, ra, rb, Ea, Eb)
        score_head = self.score_head(Ea, Eb, ra, rb, ta, tb)
        factors = self.get_factors(ha, hb, ra, rb, ta, tb)

        loss = self.loss_fn(score_tail, t_idx) + self.loss_fn(score_head, h_idx) + self.regularizer(factors)
        return loss

    def scoring_head(self, r_idx, t_idx):
        r_idx = r_idx.view(-1)
        t_idx = t_idx.view(-1)
        ra, rb = self.Ra(r_idx), self.Rb(r_idx)
        ta, tb = self.Ea(t_idx), self.Eb(t_idx)
        Ea, Eb = self.Ea.weight, self.Eb.weight
        score = self.score_head(Ea, Eb, ra, rb, ta, tb)
        return score

    def scoring_tail(self, h_idx, r_idx):
        h_idx = h_idx.view(-1)
        r_idx = r_idx.view(-1)
        ha, hb = self.Ea(h_idx), self.Eb(h_idx)
        ra, rb = self.Ra(r_idx), self.Rb(r_idx)
        Ea, Eb = self.Ea.weight, self.Eb.weight
        score = self.score_tail(ha, hb, ra, rb, Ea, Eb)
        return score

    def score_head(self, ha, hb, ra, rb, ta, tb):
        score_1 = (ta * ra + tb * rb) @ ha.transpose(-1, -2)
        score_2 = (tb * ra - ta * rb) @ hb.transpose(-1, -2)
        return score_1 + score_2

    def score_tail(self, ha, hb, ra, rb, ta, tb):
        score_1 = (ha * ra - hb * rb) @ ta.transpose(-1, -2)
        score_2 = (ha * rb + hb * ra) @ tb.transpose(-1, -2)
        return score_1 + score_2

    def get_factors(self, ha, hb, ra, rb, ta, tb) -> List[torch.Tensor]:
        factors = [
            torch.sqrt(ha ** 2 + hb ** 2),
            torch.sqrt(ra ** 2 + rb ** 2),
            torch.sqrt(ta ** 2 + tb ** 2)
        ]
        return factors


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: RelationalTripletData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, sampling_window_size, label_smoothing,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, amsgrad, lr_decay, weight_decay,
                 edim, rdim, input_dropout, hidden_dropout,
                 ):
        super(MyExperiment, self).__init__(output, local_rank=0)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        hyper = {
            'learning_rate': lr,
            'batch_size': batch_size,
            "edim": edim,
            "rdim": rdim,
        }
        self.metric_log_store.add_hyper(hyper)
        for k, v in hyper.items():
            self.log(f'{k} = {v}')
        self.metric_log_store.add_progress(max_steps)

        data.load_cache(["train_triples_ids", "test_triples_ids", "valid_triples_ids", "all_triples_ids"])
        data.load_cache(["hr_t_train"])
        data.print(self.log)
        max_relation_id = data.relation_count

        # 1. build train dataset
        train_triples, _, _ = with_inverse_relations(data.train_triples_ids, max_relation_id)
        train_data = TripleDataset(train_triples)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # 2. build valid and test dataset
        all_triples, _, _ = with_inverse_relations(data.all_triples_ids, max_relation_id)
        tail_type_constraint = defaultdict(set)
        for h, r, t in all_triples:
            tail_type_constraint[r].add(t)
        hr_t = build_map_hr_t(all_triples)
        valid_data = LinkPredictDataset(data.valid_triples_ids, hr_t, max_relation_id, data.entity_count)
        test_data = LinkPredictDataset(data.test_triples_ids, hr_t, max_relation_id, data.entity_count)
        test_type_constraint_data = LinkPredictTypeConstraintDataset(data.test_triples_ids, tail_type_constraint, hr_t, max_relation_id, data.entity_count)
        valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_type_constraint_dataloader = DataLoader(test_type_constraint_data, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # 3. build model
        model = ComplEx(data.entity_count, data.relation_count * 2, edim, rdim).to(train_device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        scheduler = get_scheduler(opt, lr_policy="step")
        best_score = 0
        best_test_score = 0
        if resume:
            if resume_by_score > 0:
                start_step, _, best_score = self.model_param_store.load_by_score(model, opt, resume_by_score)
            else:
                start_step, _, best_score = self.model_param_store.load_best(model, opt)
            self.dump_model(model)
            model.eval()
            with torch.no_grad():
                self.debug("Resumed from score %.4f." % best_score)
                self.debug("Take a look at the performance after resumed.")
                self.debug("Validation (step: %d):" % start_step)
                result = self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
                best_score = self.visual_result(start_step + 1, result, "Valid")
                self.debug("Test (step: %d):" % start_step)
                result = self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
                self.evaluate_with_type_constraint(model, test_type_constraint_data, test_type_constraint_dataloader, test_batch_size, test_device)
                best_test_score = self.visual_result(start_step + 1, result, "Test")
        else:
            model.init()
            self.dump_model(model)

        # 4. training
        self.debug("training")
        progbar = Progbar(max_step=max_steps)
        for step in range(start_step, max_steps):
            model.train()
            losses = []
            progbar_step = Progbar(max_step=len(train_dataloader))
            for idx, (h, r, t) in enumerate(train_dataloader):
                h = h.to(train_device)
                r = r.to(train_device)
                t = t.to(train_device)

                loss = model(h, r, t)

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                progbar_step.update(idx + 1, [("idx", idx + 1), ("loss", loss.item())])
            scheduler.step()

            log = {
                "loss": torch.mean(torch.Tensor(losses)).item(),
                "lr": torch.mean(torch.Tensor(scheduler.get_last_lr())).item(),
            }

            progbar.update(step + 1, [("step", step + 1), ("loss", log["loss"]), ("lr", log["lr"])])
            print("")
            for metric in log:
                self.vis.add_scalar(metric, log[metric], step)
            self.metric_log_store.add_loss(log, step + 1)

            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
                    score = self.visual_result(step + 1, result, "Valid")
                    if score >= best_score:
                        self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                        best_score = score
                        self.debug("saving best score %.4f" % score)
                        self.metric_log_store.add_best_metric({"result": result}, "Valid")
                        self.model_param_store.save_best(model, opt, step, 0, score)
                    else:
                        self.model_param_store.save_by_score(model, opt, step, 0, score)
                        self.fail("current score=%.4f < best score=%.4f" % (score, best_score))
            if (step + 1) % every_test_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Test (step: %d):" % (step + 1))
                    result = self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
                    self.evaluate_with_type_constraint(model, test_type_constraint_data, test_type_constraint_dataloader, test_batch_size, test_device)
                    score = self.visual_result(step + 1, result, "Test")
                    if score >= best_test_score:
                        best_test_score = score
                        self.metric_log_store.add_best_metric({"result": result}, "Test")
                    print("")
        # 5. report the best
        start_step, _, best_score = self.model_param_store.load_best(model, opt)
        model.eval()
        with torch.no_grad():
            self.debug("Reporting the best performance...")
            self.debug("Resumed from score %.4f." % best_score)
            self.debug("Validation (step: %d):" % start_step)
            self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
            self.debug("Test (step: %d):" % start_step)
            self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
            self.final_result = self.evaluate_with_type_constraint(model, test_type_constraint_data, test_type_constraint_dataloader, test_batch_size, test_device)
        self.metric_log_store.finish()

    def evaluate_with_type_constraint(self, model, test_data, test_dataloader, test_batch_size, device="cuda:0"):
        self.log("with type constraint")
        data = iter(test_dataloader)

        def predict(i):
            h, r, mask_for_hr, t, reverse_r, mask_for_tReverser = next(data)
            h = h.to(device)
            r = r.to(device)
            mask_for_hr = mask_for_hr.to(device)
            t = t.to(device)
            reverse_r = reverse_r.to(device)
            mask_for_tReverser = mask_for_tReverser.to(device)
            pred_tail = model.scoring_tail(h, r)
            pred_head = model.scoring_tail(t, reverse_r)
            # pred_tail = (pred_tail[0] + pred_tail[1] + pred_tail[2] + pred_tail[3]) / 2
            # pred_head = (pred_head[0] + pred_head[1] + pred_head[2] + pred_head[3]) / 2
            return t, h, pred_tail, pred_head, mask_for_hr, mask_for_tReverser

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 5))

        def log(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
            if i % (test_batch_size * 5) == 0:
                progbar.update(i // (test_batch_size * 5), [("Hits @10", np.mean(hits[9]))])

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict_type_constraint(test_batch_size, len(test_data), predict, log)
        result = as_result_dict2((hits, hits_left, hits_right, ranks, ranks_left, ranks_right))
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}    left: {2:2.2%}    right: {3:2.2%}'.format(i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))
        self.log('Mean rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
        self.log('Mean reciprocal rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))
        return result

    def evaluate(self, model, test_data, test_dataloader, test_batch_size, device="cuda:0"):
        self.log("without type constraint")
        data = iter(test_dataloader)

        def predict(i):
            h, r, mask_for_hr, t, reverse_r, mask_for_tReverser = next(data)
            h = h.to(device)
            r = r.to(device)
            mask_for_hr = mask_for_hr.to(device)
            t = t.to(device)
            reverse_r = reverse_r.to(device)
            mask_for_tReverser = mask_for_tReverser.to(device)
            pred_tail = model.scoring_tail(h, r)
            pred_head = model.scoring_tail(t, reverse_r)
            # pred_tail = (pred_tail[0] + pred_tail[1] + pred_tail[2] + pred_tail[3]) / 2
            # pred_head = (pred_head[0] + pred_head[1] + pred_head[2] + pred_head[3]) / 2
            return t, h, pred_tail, pred_head, mask_for_hr, mask_for_tReverser

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 5))

        def log(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
            if i % (test_batch_size * 5) == 0:
                progbar.update(i // (test_batch_size * 5), [("Hits @10", np.mean(hits[9]))])

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict2(test_batch_size, len(test_data), predict, log)
        result = as_result_dict2((hits, hits_left, hits_right, ranks, ranks_left, ranks_right))
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}    left: {2:2.2%}    right: {3:2.2%}'.format(i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))
        self.log('Mean rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
        self.log('Mean reciprocal rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))
        return result

    def visual_result(self, step_num: int, result, scope: str):
        average = result["average"]
        left2right = result["left2right"]
        right2left = result["right2left"]
        sorted(average)
        sorted(left2right)
        sorted(right2left)
        for i in average:
            self.vis.add_scalar(scope + i, average[i], step_num)
        for i in left2right:
            self.vis.add_scalar(scope + i, left2right[i], step_num)
        for i in right2left:
            self.vis.add_scalar(scope + i, right2left[i], step_num)
        score = get_score(result)
        return score


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
@click.option("--name", type=str, default="TuckER", help="Name of the experiment.")
@click.option("--times", type=int, default=1, help="Run multi times to get error bars.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=200, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=5, help="Number of steps.")
@click.option("--batch_size", type=int, default=128, help="Batch size.")
@click.option("--test_batch_size", type=int, default=64, help="Test batch size.")
@click.option("--sampling_window_size", type=int, default=1000, help="Sampling window size.")
@click.option("--label_smoothing", type=float, default=0.1, help="Amount of label smoothing.")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.001, help="Learning rate.")
@click.option("--amsgrad", type=bool, default=False, help="AMSGrad for Adam.")
@click.option("--lr_decay", type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
@click.option('--weight_decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
@click.option("--edim", type=int, default=200, help="Entity embedding dimensionality.")
@click.option("--rdim", type=int, default=200, help="Relation embedding dimensionality.")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option("--hidden_dropout", type=float, default=0.1, help="Dropout after the first hidden layer.")
def main(data_home, dataset, name, times,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, sampling_window_size, label_smoothing,
         train_device, test_device,
         resume, resume_by_score,
         lr, amsgrad, lr_decay, weight_decay,
         edim, rdim, input_dropout, hidden_dropout,
         ):
    output = OutputSchema(dataset + "-" + name)
    if dataset == "all":
        datasets = [get_dataset(i, data_home) for i in ["FB15k", "FB15k-237", "WN18", "WN18RR"]]
    else:
        datasets = [get_dataset(dataset, data_home)]

    for i in datasets:
        dataset = i
        cache = RelationalTripletDatasetCachePath(dataset.cache_path)
        data = RelationalTripletData(dataset=dataset, cache_path=cache)
        data.preprocess_data_if_needed()
        data.load_cache(["meta"])

        result_bracket = []
        for idx in range(times):
            seed = 10 ** idx
            set_seeds(seed)
            print("seed = ", seed)
            exp = MyExperiment(
                output, data,
                start_step, max_steps, every_test_step, every_valid_step,
                batch_size, test_batch_size, sampling_window_size, label_smoothing,
                train_device, test_device,
                resume, resume_by_score,
                lr, amsgrad, lr_decay, weight_decay,
                edim, rdim, input_dropout, hidden_dropout,
            )
            result_bracket.append(exp.final_result["average"])

        keys = list(result_bracket[0].keys())
        matrix = [[avg[key] for key in keys] for avg in result_bracket]
        result_tensor = torch.Tensor(matrix)
        result_mean = torch.mean(result_tensor, dim=0)
        result_var = torch.var(result_tensor, dim=0)
        for idx, key in enumerate(keys):
            output.logger.info(key + "  mean=" + str(float(result_mean[idx])) + "  var=" + str(float(result_var[idx])))


if __name__ == '__main__':
    main()
