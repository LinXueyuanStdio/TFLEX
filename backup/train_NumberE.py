import random

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from toolbox.evaluate.Evaluate import get_score
from toolbox.evaluate.LinkPredict import batch_link_predict2, as_result_dict
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds

max_len_of_list_of_01 = 32


def num_to_list_of_01(num):
    return list("".join(['{:08b}'.format(i) for i in num.to_bytes(length=max_len_of_list_of_01, byteorder='little', signed=True)]))


def index_list_of_ones(list_of_01):
    return [i for i, n in enumerate(list_of_01) if n == "1"]


def fixed_length_index_list_of_ones(list_of_01):
    # 0 : none, placeholder
    # 1 : first num, id instead of index. the index of id=1 is 0 in list_of_01
    # 2 : second num, id instead of index. the index of id=2 is 1 in list_of_01
    return [i + 1 if n == "1" else 0 for i, n in enumerate(list_of_01)]


class AddDataset(Dataset):
    def __init__(self, min_num: int, max_num: int):
        self.min_num = min_num
        self.max_num = max_num

    def __len__(self):
        return self.max_num - self.min_num

    def __getitem__(self, idx):
        a = random.randint(self.min_num, self.max_num)
        b = random.randint(self.min_num, self.max_num)
        c = a + b
        idx_list_a = fixed_length_index_list_of_ones(num_to_list_of_01(a))
        idx_list_b = fixed_length_index_list_of_ones(num_to_list_of_01(b))
        idx_list_c = fixed_length_index_list_of_ones(num_to_list_of_01(c))
        h = torch.LongTensor(idx_list_a)
        r = torch.LongTensor(idx_list_b)
        t = torch.LongTensor(idx_list_c)
        return h, r, t


class NumberE(nn.Module):
    def __init__(self, num, embedding_dim):
        """
        num:int 寄存器长度
        """
        super().__init__()
        self.embedding = nn.Embedding(num, embedding_dim)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.embedding(h_idx)  # BxLxd
        r = self.embedding(r_idx)  # BxLxd
        t = self.embedding(t_idx)  # BxLxd
        hr = h + r
        nn.Loss


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, sampling_window_size, label_smoothing,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, amsgrad, lr_decay, weight_decay,
                 edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
                 ):
        super(MyExperiment, self).__init__(output)
        self.log(f"{locals()}")

        # 1. build train dataset
        train_data = AddDataset(-100, 100)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        # 2. build valid and test dataset
        valid_data = AddDataset(-500, 500)
        test_data = AddDataset(-1000, 1000)
        valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # 3. build model
        model = NumberE(max_len_of_list_of_01 + 1, edim).to(train_device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        best_score = 0
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
                self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
                self.debug("Test (step: %d):" % start_step)
                self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
        else:
            model.init()
            self.dump_model(model)

        # 4. training
        progbar = Progbar(max_step=max_steps)
        for step in range(start_step, max_steps):
            model.train()
            for h, r, targets in train_dataloader:
                opt.zero_grad()

                h = h.to(train_device)
                r = r.to(train_device)
                targets = targets.to(train_device).float()

                predictions = model(h, r)
                loss = model.loss(predictions, targets)
                # loss = loss + model.regular_loss(h, r)
                loss.backward()
                opt.step()

            progbar.update(step + 1, [("step", step + 1), ("loss", loss.item())])
            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
                    self.visual_result(step + 1, result, "Valid-")
                    score = get_score(result)
                    if score >= best_score:
                        self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                        best_score = score
                        self.debug("saving best score %.4f" % score)
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
                    self.visual_result(step + 1, result, "Test-")

    def evaluate(self, model, test_data, test_dataloader, test_batch_size, device="cuda:0"):
        data = iter(test_dataloader)

        def predict(i):
            h, r, mask_for_hr, t, reverse_r, mask_for_tReverser = next(data)
            h = h.to(device)
            r = r.to(device)
            mask_for_hr = mask_for_hr.to(device)
            t = t.to(device)
            reverse_r = reverse_r.to(device)
            mask_for_tReverser = mask_for_tReverser.to(device)
            pred1 = model(h, r)
            pred2 = model(t, reverse_r)
            pred1 = pred1[0] + pred1[1]
            pred2 = pred2[0] + pred2[1]
            return t, h, pred1, pred2, mask_for_hr, mask_for_tReverser

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 10))

        def log(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
            if i % (test_batch_size * 10) == 0:
                progbar.update(i // (test_batch_size * 10), [("Hits @10", np.mean(hits[9]))])

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict2(test_batch_size, len(test_data), predict, log)
        result = as_result_dict((hits, hits_left, hits_right, ranks, ranks_left, ranks_right))
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


@click.command()
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
@click.option("--name", type=str, default="NumberE", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=1000, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=5, help="Number of steps.")
@click.option("--batch_size", type=int, default=512, help="Batch size.")
@click.option("--test_batch_size", type=int, default=512, help="Test batch size.")
@click.option("--sampling_window_size", type=int, default=1000, help="Sampling window size.")
@click.option("--label_smoothing", type=float, default=0.1, help="Amount of label smoothing.")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.003, help="Learning rate.")
@click.option("--amsgrad", type=bool, default=False, help="AMSGrad for Adam.")
@click.option("--lr_decay", type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
@click.option('--weight_decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
@click.option("--edim", type=int, default=200, help="Entity embedding dimensionality.")
@click.option("--rdim", type=int, default=200, help="Relation embedding dimensionality.")
@click.option("--input_dropout", type=float, default=0.2, help="Input layer dropout.")
@click.option("--hidden_dropout1", type=float, default=0.2, help="Dropout after the first hidden layer.")
@click.option("--hidden_dropout2", type=float, default=0.2, help="Dropout after the second hidden layer.")
def main(dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, sampling_window_size, label_smoothing,
         train_device, test_device,
         resume, resume_by_score,
         lr, amsgrad, lr_decay, weight_decay,
         edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
         ):
    set_seeds()
    output = OutputSchema(dataset + "-" + name)

    MyExperiment(
        output,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, sampling_window_size, label_smoothing,
        train_device, test_device,
        resume, resume_by_score,
        lr, amsgrad, lr_decay, weight_decay,
        edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
    )


if __name__ == '__main__':
    main()
