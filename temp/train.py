from collections import defaultdict

import numpy as np
import torch

from temp.metrics import get_ranks, compute_hits, compute_MRR


def get_ert_vocab(data):
    """
    Construct a dict of the data containing [E,R,T] as keys and target entities as values
    """

    ert_vocab = defaultdict(list)
    for quad in data:
        ert_vocab[(quad[0], quad[1], quad[2])].append(quad[3])
        ert_vocab[(quad[3], quad[1], quad[2])].append(quad[0])  # reverse fact
    return ert_vocab


def get_batch(batch_size, ert_vocab, ert_vocab_pairs, idx, n_entities, device='cpu'):
    """
    Return a batch of data for training

    Parameters
    ----------

    batch_size : int,
        .
    ert_vocab : dict,
        Dict containing [E,R,T] as keys and target entities as values
    ert_vocab_pairs : list,
        list of ert_vocab keys
    idx: int,
        Batch number
    n_entities : int,
        Total number of entities considered in the model (n_e)
    device :  {'cpu','cuda'}
        On which device to do the computation
    """

    batch = ert_vocab_pairs[idx:idx + batch_size]
    targets = np.zeros((len(batch), n_entities))

    for idx, pair in enumerate(batch):
        targets[idx, ert_vocab[pair]] = 1.
    targets = torch.FloatTensor(targets).to(device)

    return np.array(batch), targets


def train_temporal(model, data, n_iter=200, learning_rate=0.0005, batch_size=128, print_loss_every=1, early_stopping=20, label_smoothing=0., device='cpu'):
    """
    Train a temporal KG model

    Parameters
    -----------
    model : TuckER instance,
        TuckER model
    data : obj from class Data:
        contains train,test,valid data in the form of idx matrices
    n_iter : int,
        Number of iterations
    learning_rate : float,
        Learning rate
    batch size : int,
        Batch size
    print_loss_every : int,
        Frequency for when to print the losses
    early_stopping : {False,int}:
        If False does nothing, if a number will perform early stopping using this int
    device : {'cpu','cuda'}
        On which device to do the computation
    """

    data_idxs, data_idxs_valid, data_idxs_test = data.train_data_idxs, data.valid_data_idxs, data.test_data_idxs

    if early_stopping == False:
        early_stopping = n_iter + 1

    model.init()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ert_vocab = get_ert_vocab(data_idxs)
    ert_vocab = {k: v for (k, v) in ert_vocab.items() if (type(k) is list or type(k) is tuple) and len(k) == 3}

    n_entities = int(max(data_idxs[:, 0]) + 1)

    ert_vocab_pairs = list(ert_vocab.keys())

    # Validation set for early stopping
    targets_valid = np.zeros((data_idxs_valid.shape[0], n_entities))
    for idx, ent_id in enumerate(data_idxs_valid[:, -1]):
        targets_valid[idx, ent_id] = 1
    targets_valid = torch.FloatTensor(targets_valid).to(device)
    if label_smoothing:
        targets_valid = ((1.0 - label_smoothing) * targets_valid) + (1.0 / targets_valid.size(1))

        # Init params
    model.train()
    losses = []
    loss_valid_all = []
    mrr = []
    hits = []

    for i in range(n_iter):
        loss_batch = []
        loss_valid = []

        for j in range(0, len(ert_vocab_pairs), batch_size):
            # get the current batch
            data_batch, targets = get_batch(batch_size, ert_vocab, ert_vocab_pairs, j, n_entities=n_entities, device=device)

            if label_smoothing:
                targets = ((1.0 - label_smoothing) * targets) + (1.0 / targets.size(1))

            opt.zero_grad()
            e1_idx = torch.tensor(data_batch[:, 0]).to(device)
            r_idx = torch.tensor(data_batch[:, 1]).to(device)
            t_idx = torch.tensor(data_batch[:, 2]).to(device)

            predictions = model.forward(e1_idx, r_idx, t_idx)
            loss = model.loss(predictions, targets)

            loss.backward()
            opt.step()

            loss_batch.append(loss.item())
        losses.append(np.mean(loss_batch))

        # Compute validation loss
        for j in range(0, len(targets_valid) // 32):
            data_j = torch.tensor(data_idxs_valid[j * 32:(j + 1) * 32]).to(device)
            pred_valid = model.forward(data_j[:, 0], data_j[:, 1], data_j[:, 2]).detach()
            loss_valid_j = model.loss(pred_valid, targets_valid[j * 32:(j + 1) * 32]).item()
            loss_valid.append(loss_valid_j)

        data_j = torch.tensor(data_idxs_valid[(j + 1) * 32:]).to(device)
        pred_valid = model.forward(data_j[:, 0], data_j[:, 1], data_j[:, 2]).detach()
        loss_valid_j = model.loss(pred_valid, targets_valid[(j + 1) * 32:]).item()
        loss_valid.append(loss_valid_j)

        loss_valid_all.append(np.mean(loss_valid))

        # Test metrics
        test_ranks = get_ranks(model, torch.tensor(data_idxs_test), torch.tensor(data_idxs_test[:, -1]), device=device)
        test_mrr = compute_MRR(test_ranks)
        test_hits1 = compute_hits(test_ranks, 1)
        test_hits3 = compute_hits(test_ranks, 3)
        test_hits10 = compute_hits(test_ranks, 10)

        mrr.append(test_mrr)
        hits.append([test_hits1, test_hits3, test_hits10])

        if i % print_loss_every == 0:
            print(f"{i + 1}/{n_iter} loss = {losses[-1]}, valid loss = {np.mean(loss_valid)}, test MRR : {test_mrr}")

        # Early Stopping 
        if i > early_stopping:
            if min(loss_valid_all[-early_stopping:]) > min(loss_valid_all) or min(loss_valid_all[-early_stopping:]) > min(loss_valid_all[:-early_stopping]) - 5e-8:
                print(f"{i}/{n_iter} loss = {losses[-1]}, valid loss = {np.mean(loss_valid)} , test MRR : {test_mrr}")

                model.eval()
                best = np.argmin(loss_valid_all)
                return model, [mrr[best], *hits[best]]

    model.eval()

    if early_stopping:
        best = np.argmin(loss_valid_all)
        return model, [mrr[best], *hits[best]]
    else:
        return model
