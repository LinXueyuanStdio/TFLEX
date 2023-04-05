from collections import defaultdict
from typing import Union, Dict, List

import numpy as np
import pandas as pd
import torch


def link_predict(predictions, truth):
    """
    predictions : torch.Tensor, similarity matrix of shape (batch_size, entity_count)
    truth: torch.Tensor, vector of length (BatchSize)
    """
    pass


def empty_log(i, hits, ranks):
    pass


def batch_link_predict(test_batch_size: int, max_iter: int, predict, log=empty_log):
    """
    predictions : torch.Tensor, similarity matrix of shape (batch_size, entity_count)
    truth: torch.Tensor, vector of length (BatchSize)
    """
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])
    for idx in range(0, max_iter, test_batch_size):
        t, predictions, truth = predict(idx)
        predictions = predictions - predictions * truth
        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()
        for i in range(t.shape[0]):
            rank = np.where(sort_idxs[i] == t[i, 0].item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
        log(idx, hits, ranks)
    return hits, ranks


def empty_log2(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
    pass


def batch_link_predict2(test_batch_size: int, max_iter: int, predict, log=empty_log2):
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for idx in range(0, max_iter, test_batch_size):
        t, h, pred1, pred2, truth1, truth2 = predict(idx)

        # filter existing other answers and leave current answer for testing
        pred1 = pred1 - pred1 * truth1
        pred2 = pred2 - pred2 * truth2

        # sort and rank
        sort_values1, sort_idxs1 = torch.sort(pred1, dim=1, descending=True)
        sort_values2, sort_idxs2 = torch.sort(pred2, dim=1, descending=True)

        sort_idxs1 = sort_idxs1.cpu().numpy()
        sort_idxs2 = sort_idxs2.cpu().numpy()
        for i in range(h.shape[0]):
            # find the rank of the target entities
            t_idx = t[i, 0].item()
            h_idx = h[i, 0].item()
            rank1 = np.where(sort_idxs1[i] == t_idx)[0][0]
            rank2 = np.where(sort_idxs2[i] == h_idx)[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)
        log(idx, hits, hits_left, hits_right, ranks, ranks_left, ranks_right)
    return hits, hits_left, hits_right, ranks, ranks_left, ranks_right


def batch_link_predict_type_constraint(test_batch_size: int, max_iter: int, predict, log=empty_log2):
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for idx in range(0, max_iter, test_batch_size):
        t, h, pred1, pred2, truth1, truth2 = predict(idx)

        # 1. type constraint
        pred1 = pred1 * truth1
        pred2 = pred2 * truth2

        # 3. sort and rank
        sort_values1, sort_idxs1 = torch.sort(pred1, 1, descending=True)
        sort_values2, sort_idxs2 = torch.sort(pred2, 1, descending=True)

        sort_idxs1 = sort_idxs1.cpu().numpy()
        sort_idxs2 = sort_idxs2.cpu().numpy()
        for i in range(h.shape[0]):
            # find the rank of the target entities
            t_idx = t[i, 0].item()
            h_idx = h[i, 0].item()
            rank1 = np.where(sort_idxs1[i] == t_idx)[0][0]
            rank2 = np.where(sort_idxs2[i] == h_idx)[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)
        log(idx, hits, hits_left, hits_right, ranks, ranks_left, ranks_right)
    return hits, hits_left, hits_right, ranks, ranks_left, ranks_right


def as_result_dict(metrics):
    """
    result = {
        "average": {
            "Hits@1": float,
            "Hits@3": float,
            "Hits@10": float,
            "MeanRank": float,
            "MeanReciprocalRank": float,
        },
    }
    """
    hits, ranks = metrics
    top_k = (0, 2, 9)
    result = {
        "average": {}
    }
    for i in top_k:
        result["average"]["Hits@{}".format(i + 1)] = np.mean(hits[i])
    result["average"]["MeanRank"] = np.mean(ranks)
    result["average"]["MeanReciprocalRank"] = np.mean(1. / np.array(ranks))
    return result


def as_result_dict2(metrics):
    """
    result = {
        "average": {
            "Hits@1": float,
            "Hits@3": float,
            "Hits@10": float,
            "MeanRank": float,
            "MeanReciprocalRank": float,
        },
        "left2right": {
            "Hits@1": float,
            "Hits@3": float,
            "Hits@10": float,
            "MeanRank": float,
            "MeanReciprocalRank": float,
        },
        "right2left": {
            "Hits@1": float,
            "Hits@3": float,
            "Hits@10": float,
            "MeanRank": float,
            "MeanReciprocalRank": float,
        },
    }
    """
    hits, hits_left, hits_right, ranks, ranks_left, ranks_right = metrics
    top_k = (0, 2, 9)
    result = {"average": {}, "left2right": {}, "right2left": {}}
    for i in top_k:
        result["average"]["Hits@{}".format(i + 1)] = np.mean(hits[i])
        result["left2right"]["Hits@{}".format(i + 1)] = np.mean(hits_left[i])
        result["right2left"]["Hits@{}".format(i + 1)] = np.mean(hits_right[i])
    result["average"]["MeanRank"] = np.mean(ranks)
    result["average"]["MeanReciprocalRank"] = np.mean(1. / np.array(ranks))
    result["left2right"]["MeanRank"] = np.mean(ranks_left)
    result["left2right"]["MeanReciprocalRank"] = np.mean(1. / np.array(ranks_left))
    result["right2left"]["MeanRank"] = np.mean(ranks_right)
    result["right2left"]["MeanReciprocalRank"] = np.mean(1. / np.array(ranks_right))
    return result


key2short = {
    "average": "avg",
    "left2right": "l2r",
    "right2left": "r2l",
    "Hits@1": "H@1",
    "Hits@3": "H@3",
    "Hits@10": "H@10",
    "MeanRank": "MR",
    "MeanReciprocalRank": "MRR",
}


def get_score(result: Dict[str, Union[dict, float]]):
    score = result["average"]["MeanReciprocalRank"]
    return score


def dataframe_from_result2(result: Dict[str, Union[dict, float]]) -> pd.DataFrame:
    df = pd.DataFrame()
    for key, value in result.items():
        df[key] = list(value.values())
    df.index = ["Hits@1", "Hits@3", "Hits@10", "MR", "MRR"]
    return df


def to_str(data):
    if isinstance(data, float):
        if data < 1:
            return "{0:>6.2%}  ".format(data)
        else:
            return "{0:>6.2f}  ".format(data)
    elif isinstance(data, int):
        return "{0:^6d}  ".format(data)
    else:
        return "{0:^6s}  ".format(data[:6])


def print_dataframe(df: pd.DataFrame, printer=print):
    for i in range(df.shape[0]):
        printer(df.iloc[i])


def pretty_print2(scope: str, result: Dict[str, Union[dict, float]], printer=print) -> Dict[str, List[Union[str, float]]]:
    header = "{0:<8s}".format(scope)
    row_results = defaultdict(list)
    for col in result:
        row_results[header].append(key2short[col] if col in key2short else col)
        col_data = result[col]
        for row in col_data:
            cell = col_data[row]
            key_row = key2short[row] if row in key2short else row
            row_results[key_row].append(cell)

    max_len = max([len(row) for row in row_results])

    for i in row_results:
        row = row_results[i]
        printer(("{0:<" + str(max_len) + "s}").format(i)[:max_len] + ": " + "".join([to_str(data) for data in row]))

    return row_results


def pretty_print(result: Dict[str, Union[dict, float]], printer=print):
    average = result["average"]
    left2right = result["left2right"]
    right2left = result["right2left"]
    sorted(average)
    sorted(left2right)
    sorted(right2left)
    printer('---------------------------')
    printer('Average:')
    for i in average:
        if i.startswith("Hit"):
            printer('%s: %.2f%%' % (i, left2right[i] * 100))
        else:
            printer('%s: %.2f' % (i, left2right[i]))
    printer('For each left:')
    for i in left2right:
        if i.startswith("Hit"):
            printer('%s: %.2f%%' % (i, left2right[i] * 100))
        else:
            printer('%s: %.2f' % (i, left2right[i]))
    printer('For each right:')
    for i in right2left:
        if i.startswith("Hit"):
            printer('%s: %.2f%%' % (i, right2left[i] * 100))
        else:
            printer('%s: %.2f' % (i, right2left[i]))
    printer('---------------------------')
