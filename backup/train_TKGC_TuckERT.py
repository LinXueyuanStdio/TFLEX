import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ComplexTemporalQueryData import TemporalKnowledgeDatasetCachePath, ICEWS05_15, ICEWS14, TemporalKnowledgeData, build_map_srt_o
from toolbox.data.LinkPredictDataset import TemporalLinkPredictDataset
from toolbox.data.ScoringAllDataset import TemporalScoringAllDataset
from toolbox.evaluate.LinkPredict import batch_link_predict, get_score, as_result_dict
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.optim.lr_scheduler import get_scheduler
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds

set_seeds()


class TuckERT(nn.Module):
    def __init__(self, ne: int, nr: int, nt: int, de: int, dr: int, dt: int,
                 input_dropout=0.1,
                 hidden_dropout1=0.1,
                 hidden_dropout2=0.1):
        super(TuckERT, self).__init__()
        # Embeddings dimensionality
        self.de = de
        self.dr = dr
        self.dt = dt

        # Data dimensionality
        self.ne = ne
        self.nr = nr
        self.nt = nt

        # Embedding matrices
        self.E = nn.Embedding(self.ne, de)
        self.R = nn.Embedding(self.nr, dr)
        self.T = nn.Embedding(self.nt, dt)

        # Core tensor
        self.W = nn.Parameter(torch.rand((dr, de, de, dt)), requires_grad=True)

        # Special layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)
        self.loss = nn.BCELoss()

        self.bne = nn.BatchNorm1d(de)

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
        nn.init.xavier_normal_(self.T.weight.data)
        nn.init.uniform_(self.W, -0.1, 0.1)

    def forward(self, e1_idx, r_idx, t_idx):
        # Mode 1 product with entity vector
        e1 = self.E(e1_idx).view(-1, self.de)
        x = self.bne(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, self.de)

        # Mode 2 product with relation vector
        r = self.R(r_idx).view(-1, self.dr)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, self.de, self.de * self.dt)
        x = torch.bmm(x, W_mat)

        # Mode 3 product with time vector
        t = self.T(t_idx).view(-1, self.dt)
        x = x.view(-1, self.de, self.dt)
        x = torch.bmm(x, t.view(*t.shape, -1))

        # Mode 4 product with entity matrix
        x = x.view(-1, self.de)
        x = torch.mm(x, self.E.weight.transpose(1, 0))

        pred = torch.sigmoid(x)
        return pred


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: TemporalKnowledgeData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, sampling_window_size, label_smoothing,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, amsgrad, lr_decay, weight_decay,
                 edim, rdim, tdim,
                 input_dropout, hidden_dropout1, hidden_dropout2,
                 ):
        super(MyExperiment, self).__init__(output, local_rank=0)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        hyper = {
            'learning_rate': lr,
            'batch_size': batch_size,
            "edim": edim,
            "rdim": rdim,
            "tdim": tdim,
            "input_dropout": input_dropout,
            "hidden_dropout1": hidden_dropout1,
            "hidden_dropout2": hidden_dropout2,
            "label_smoothing": label_smoothing,
        }
        self.metric_log_store.add_hyper(hyper)
        for k, v in hyper.items():
            self.log(f'{k} = {v}')
        self.metric_log_store.add_progress(max_steps)

        data.load_cache(["train_triples_ids", "test_triples_ids", "valid_triples_ids", "all_triples_ids"])
        data.print(self.log)

        # 1. build train dataset
        # train_triples, _, _ = with_inverse_relations(data.train_triples_ids, data.relation_count)
        # all_triples, _, _ = with_inverse_relations(data.all_triples_ids, data.relation_count)

        train_triples = data.train_triples_ids
        all_triples = data.all_triples_ids

        train_data = TemporalScoringAllDataset(train_triples, data.entity_count)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        srt_o = build_map_srt_o(all_triples)

        # 2. build valid and test dataset
        valid_data = TemporalLinkPredictDataset(data.valid_triples_ids, srt_o, data.entity_count)
        test_data = TemporalLinkPredictDataset(data.test_triples_ids, srt_o, data.entity_count)
        valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # 3. build model
        model = TuckERT(data.entity_count, data.relation_count, data.timestamp_count, edim, rdim, tdim, input_dropout, hidden_dropout1, hidden_dropout2).to(train_device)
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
            for s, r, t, targets in train_dataloader:
                opt.zero_grad()

                s = s.to(train_device)
                r = r.to(train_device)
                t = t.to(train_device)
                targets = targets.to(train_device).float()
                if label_smoothing:
                    targets = ((1.0 - label_smoothing) * targets) + (1.0 / targets.size(1))

                predictions = model(s, r, t).float()
                loss = model.loss(predictions, targets)
                # print(loss)
                # loss = loss + model.regular_loss(h, r)
                losses.append(loss.item())
                loss.backward()
                opt.step()
            # scheduler.step(step + 1)
            scheduler.step()

            log = {
                "loss": torch.mean(torch.Tensor(losses)).item(),
                "lr": torch.mean(torch.Tensor(scheduler.get_last_lr())).item(),
            }

            progbar.update(step + 1, [("step", step + 1), ("loss", log["loss"]), ("lr", log["lr"])])
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
            self.final_result = self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
        self.metric_log_store.finish()

    def evaluate(self, model, test_data, test_dataloader, test_batch_size, device="cuda:0"):
        self.log("without type constraint")
        data = iter(test_dataloader)

        def predict(i):
            s, r, t, o, mask_for_srt = next(data)
            s = s.to(device)
            r = r.to(device)
            t = t.to(device)
            o = o.to(device)
            mask_for_srt = mask_for_srt.to(device)
            pred = model(s, r, t).float()
            return o, pred, mask_for_srt

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 10))

        def log(i, hits, ranks):
            if i % (test_batch_size * 10) == 0:
                progbar.update(i // (test_batch_size * 10), [("Hits @10", np.mean(hits[9]))])

        hits, ranks = batch_link_predict(test_batch_size, len(test_data), predict, log)
        result = as_result_dict((hits, ranks))
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}'.format(i + 1, np.mean(hits[i])))
        self.log('Mean rank: {0:.3f}'.format(np.mean(ranks)))
        self.log('Mean reciprocal rank: {0:.3f}'.format(np.mean(1. / np.array(ranks))))
        return result

    def visual_result(self, step_num: int, result, scope: str):
        average = result["average"]
        sorted(average)
        for i in average:
            self.vis.add_scalar(scope + i, average[i], step_num)
        score = get_score(result)
        return score


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="ICEWS14", help="Which dataset to use: ICEWS14, ICEWS05_15.")
@click.option("--name", type=str, default="TuckERT", help="Name of the experiment.")
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
@click.option("--tdim", type=int, default=200, help="Timestamp embedding dimensionality.")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option("--hidden_dropout1", type=float, default=0.1, help="Dropout after the first hidden layer.")
@click.option("--hidden_dropout2", type=float, default=0.1, help="Dropout after the second hidden layer.")
def main(data_home, dataset, name, times,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, sampling_window_size, label_smoothing,
         train_device, test_device,
         resume, resume_by_score,
         lr, amsgrad, lr_decay, weight_decay,
         edim, rdim, tdim,
         input_dropout, hidden_dropout1, hidden_dropout2,
         ):
    output = OutputSchema(dataset + "-" + name)

    if dataset == "ICEWS14":
        dataset = ICEWS14(data_home)
    elif dataset == "ICEWS05_15":
        dataset = ICEWS05_15(data_home)
    cache = TemporalKnowledgeDatasetCachePath(dataset.cache_path)
    data = TemporalKnowledgeData(dataset=dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache(["meta", "all_triples_ids", "train_triples_ids", "test_triples_ids", "valid_triples_ids"])


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
            edim, rdim, tdim,
            input_dropout, hidden_dropout1, hidden_dropout2,
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
