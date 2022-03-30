import numpy as np
import torch


def get_ranks(model, torch_data_idxs, targets, batch_size=128, device='cpu'):
    """
    Compute ranks

    model : torch.nn.Module,
        model from which to do the prediction
    torch_data_idxs : torch.tensor,
        data matrix
    targets : array,
        List of targets values/labels
    batch_size : int,
        .
    device : {'cpu','cuda'}
        On which device to do the computation
    """
    ranks = []

    for i in range(torch_data_idxs.shape[0] // batch_size):
        data_batch = torch_data_idxs[i * batch_size:(i + 1) * batch_size].to(device)
        predictions = model.forward(data_batch[:, 0], data_batch[:, 1], data_batch[:, 2]).detach().cpu()

        _, sort_idxs = torch.sort(predictions, dim=1, descending=True)
        for j in range(predictions.shape[0]):
            rank = np.where(sort_idxs[j] == targets[i * batch_size + j])[0][0]
            ranks.append(rank + 1)

    data_batch = torch_data_idxs[(i + 1) * batch_size:].to(device)
    predictions = model.forward(data_batch[:, 0], data_batch[:, 1], data_batch[:, 2]).detach().cpu()

    _, sort_idxs = torch.sort(predictions, dim=1, descending=True)
    for j in range(predictions.shape[0]):
        rank = np.where(sort_idxs[j] == targets[(i + 1) * batch_size + j])[0][0]
        ranks.append(rank + 1)

    return ranks


def compute_MRR(ranks):
    """Compute the Mean Reciprocal Rank"""
    return np.mean(1 / np.array(ranks))


def compute_hits(ranks, n):
    """Compute Hits@n"""
    return len(np.where(np.array(ranks) <= n)[0]) / len(ranks)
