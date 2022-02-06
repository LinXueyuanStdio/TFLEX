import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from toolbox.data.DataSchema import RelationalTripletData, RelationalTripletDatasetCachePath
from toolbox.data.DatasetSchema import FreebaseFB15k_237
from toolbox.data.LinkPredictDataset import LinkPredictDataset
from toolbox.data.ScoringAllDataset import ScoringAllDataset
from toolbox.data.functional import with_inverse_relations, build_map_hr_t
from toolbox.evaluate.LinkPredict import batch_link_predict2
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.nn.ConvE import ConvE
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds

set_seeds()


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: RelationalTripletData,
                 max_steps, every_test_step, every_valid_step, batch_size, test_batch_size,
                 lr, amsgrad, lr_decay, weight_decay, edim, rdim, train_device, test_device, sampling_window_size,
                 input_dropout, hidden_dropout1, hidden_dropout2, label_smoothing,
                 ):
        super(MyExperiment, self).__init__(output)
        data.load_cache(["train_triples_ids", "test_triples_ids", "valid_triples_ids", "all_triples_ids"])
        data.print(self.log)

        train_triples, _, _ = with_inverse_relations(data.train_triples_ids, data.relation_count)
        all_triples, _, _ = with_inverse_relations(data.all_triples_ids, data.relation_count)

        train_data = ScoringAllDataset(train_triples, data.entity_count)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        hr_t = build_map_hr_t(all_triples)
        valid_data = LinkPredictDataset(data.valid_triples_ids, hr_t, data.relation_count, data.entity_count)
        test_data = LinkPredictDataset(data.test_triples_ids, hr_t, data.relation_count, data.entity_count)
        valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        model = ConvE(data.entity_count, 2 * data.relation_count, edim, hidden_dropout=hidden_dropout1).to(train_device)
        model.init()
        self.log(model)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        progbar = Progbar(max_step=max_steps)
        for step in range(max_steps):
            model.train()
            for h, r, targets in train_dataloader:
                opt.zero_grad()

                h = h.to(train_device)
                r = r.to(train_device)
                targets = targets.to(train_device).float()

                predictions = model(h, r).float()

                if label_smoothing:
                    targets = ((1.0 - label_smoothing) * targets) + (1.0 / targets.size(1))

                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()

            progbar.update(step + 1, [("step", step + 1), ("loss", loss.item())])
            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    self.log("")
                    self.log("Validation:")
                    self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
            if (step + 1) % every_test_step == 0:
                model.eval()
                with torch.no_grad():
                    self.log("")
                    self.log("Test:")
                    self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)

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
            pred1 = model(h, r).float()
            pred2 = model(t, reverse_r).float()
            return t, h, pred1, pred2, mask_for_hr, mask_for_tReverser

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 10))

        def log(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
            if i % (test_batch_size * 10) == 0:
                progbar.update(i // (test_batch_size * 10), [("Hits @10", np.mean(hits[9]))])

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict2(test_batch_size, len(test_data), predict, log)
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}    left: {2:2.2%}    right: {3:2.2%}'.format(i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))
        self.log('Mean rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
        self.log('Mean reciprocal rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))


@click.command()
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
@click.option("--name", type=str, default="Echo", help="Name of the experiment.")
@click.option("--max_steps", type=int, default=1000, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=5, help="Number of steps.")
@click.option("--batch_size", type=int, default=512, help="Batch size.")
@click.option("--test_batch_size", type=int, default=512, help="Test batch size.")
@click.option("--lr", type=float, default=0.003, help="Learning rate.")
@click.option("--amsgrad", type=bool, default=False, help="AMSGrad for Adam.")
@click.option("--lr_decay", type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
@click.option('--weight_decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
@click.option("--edim", type=int, default=200, help="Entity embedding dimensionality.")
@click.option("--rdim", type=int, default=200, help="Relation embedding dimensionality.")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--sampling_window_size", type=int, default=1000, help="Sampling window size.")
@click.option("--input_dropout", type=float, default=0.2, help="Input layer dropout.")
@click.option("--hidden_dropout1", type=float, default=0.3, help="Dropout after the first hidden layer.")
@click.option("--hidden_dropout2", type=float, default=0.2, help="Dropout after the second hidden layer.")
@click.option("--label_smoothing", type=float, default=0.1, help="Amount of label smoothing.")
def main(dataset, name,
         max_steps, every_test_step, every_valid_step, batch_size, test_batch_size,
         lr, amsgrad, lr_decay, weight_decay, edim, rdim, train_device, test_device, sampling_window_size,
         input_dropout, hidden_dropout1, hidden_dropout2, label_smoothing,
         ):
    output = OutputSchema(dataset + "-" + name)

    dataset = FreebaseFB15k_237()
    cache = RelationalTripletDatasetCachePath(dataset.cache_path)
    data = RelationalTripletData(dataset=dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache(["meta"])

    MyExperiment(
        output, data,
        max_steps, every_test_step, every_valid_step, batch_size, test_batch_size,
        lr, amsgrad, lr_decay, weight_decay, edim, rdim, train_device, test_device, sampling_window_size,
        input_dropout, hidden_dropout1, hidden_dropout2, label_smoothing,
    )


if __name__ == '__main__':
    main()
