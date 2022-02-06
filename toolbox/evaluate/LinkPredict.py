import numpy as np
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
    for i in range(0, max_iter, test_batch_size):
        predictions, truth = predict(i)
        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

        sort_idxs = sort_idxs.cpu().numpy()
        for j in range(truth.shape[0]):
            rank = np.where(sort_idxs[j] == truth[j].item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
        log(i, hits, ranks)
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
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(h.shape[0]):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i] == t[i, 0].item())[0][0]
            rank2 = np.where(argsort2[i] == h[i, 0].item())[0][0]
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
        result["average"]["Hits@{}".format(i+1)] = np.mean(hits[i])
        result["left2right"]["Hits@{}".format(i+1)] = np.mean(hits_left[i])
        result["right2left"]["Hits@{}".format(i+1)] = np.mean(hits_right[i])
    result["average"]["MeanRank"] = np.mean(ranks)
    result["average"]["MeanReciprocalRank"] = np.mean(1. / np.array(ranks))
    result["left2right"]["MeanRank"] = np.mean(ranks_left)
    result["left2right"]["MeanReciprocalRank"] = np.mean(1. / np.array(ranks_left))
    result["right2left"]["MeanRank"] = np.mean(ranks_right)
    result["right2left"]["MeanReciprocalRank"] = np.mean(1. / np.array(ranks_right))
    return result


def pretty_print(result, printer=print):
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


