# 指标计算
#
#
# outline
# 1. utils function
# 2. exported function

import time
import timeit
from typing import Dict, Union

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn import preprocessing

# region 1. utils function
from toolbox.utils.Progbar import Progbar


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def multi_cal_rank(task, distance, distanceT, top_k):
    acc_l2r, acc_r2l = np.array([0.] * len(top_k)), np.array([0.] * len(top_k))
    mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
    for i in range(len(task)):
        ref = task[i]
        indices = distance[i, :].argsort()
        rank = np.where(indices == ref)[0][0]
        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for j in range(len(top_k)):
            if rank < top_k[j]:
                acc_l2r[j] += 1
    for i in range(len(task)):
        ref = task[i]
        indices = distanceT[:, i].argsort()
        rank = np.where(indices == ref)[0][0]
        mean_r2l += (rank + 1)
        mrr_r2l += 1.0 / (rank + 1)
        for j in range(len(top_k)):
            if rank < top_k[j]:
                acc_r2l[j] += 1
    return acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l


def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : np.array, An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'euclidean':
        sim_mat = 1 - cdist(embed1, embed2, metric='euclidean')  # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')  # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : np.array, A csls similarity matrix of n1*n2.
    """
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)


# endregion


# region 2. exported function
def evaluate_entity_alignment(test_data_size: int, left_emb, right_emb,
                              rerank=False,
                              metric="euclidean",
                              csls=0) -> Dict[str, Union[dict, float]]:
    """
    evaluate metrics

    :param test_data_size: (int) test_data_size = len(test), here test = [(int, int)] is entity alignment id pair list
    :param left_emb: left entity embedding, test_data_size x d, where d is embedding dimension
    :param right_emb: right entity embedding, test_data_size x d
    :param rerank: (False)
    :param metric: distance between left entity and right entity
    :param csls: (0)

    result = {
        "left2right": {
            "Hits@1": float,
            "Hits@3": float,
            "Hits@5": float,
            "Hits@10": float,
            "Hits@50": float,
            "Hits@100": float,
            "MeanRank": float,
            "MeanReciprocalRank": float,
        },
        "right2left": {
            "Hits@1": float,
            "Hits@3": float,
            "Hits@5": float,
            "Hits@10": float,
            "Hits@50": float,
            "Hits@100": float,
            "MeanRank": float,
            "MeanReciprocalRank": float,
        },
        "time": float (seconds)
    }
    """
    t_test = time.time()
    top_k = [1, 3, 5, 10, 50, 100]

    distance = - sim(left_emb, right_emb, metric=metric, normalize=True, csls_k=csls)
    if rerank:
        indices = np.argsort(np.argsort(distance, axis=1), axis=1)
        indices_ = np.argsort(np.argsort(distance.T, axis=1), axis=1)
        distance = indices + indices_.T

    tasks = div_list(np.array(range(test_data_size)), 20)

    acc_l2r, acc_r2l = np.array([0.] * len(top_k)), np.array([0.] * len(top_k))
    mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
    for task in tasks:
        res = multi_cal_rank(task, distance[task, :], distance[:, task], top_k)
        (_acc_l2r, _mean_l2r, _mrr_l2r, _acc_r2l, _mean_r2l, _mrr_r2l) = res
        acc_l2r += _acc_l2r
        mean_l2r += _mean_l2r
        mrr_l2r += _mrr_l2r
        acc_r2l += _acc_r2l
        mean_r2l += _mean_r2l
        mrr_r2l += _mrr_r2l
    mean_l2r /= test_data_size
    mean_r2l /= test_data_size
    mrr_l2r /= test_data_size
    mrr_r2l /= test_data_size
    for i in range(len(top_k)):
        acc_l2r[i] = round(acc_l2r[i] / test_data_size, 4)
        acc_r2l[i] = round(acc_r2l[i] / test_data_size, 4)

    result = {"left2right": {}, "right2left": {}, "time": time.time() - t_test}
    for i, k in enumerate(top_k):
        result["left2right"]["Hits@{}".format(k)] = acc_l2r[i]
        result["right2left"]["Hits@{}".format(k)] = acc_r2l[i]
    result["left2right"]["MeanRank"] = mean_l2r
    result["left2right"]["MeanReciprocalRank"] = mrr_l2r
    result["right2left"]["MeanRank"] = mean_r2l
    result["right2left"]["MeanReciprocalRank"] = mrr_r2l
    return result


def get_score(result: Dict[str, Union[dict, float]]):
    score = (result["left2right"]["MeanReciprocalRank"] + result["right2left"]["MeanReciprocalRank"]) / 2
    return score


def pretty_print(result, printer=print):
    left2right = result["left2right"]
    right2left = result["right2left"]
    using_time = result["time"]
    sorted(left2right)
    sorted(right2left)
    printer('---------------------------')
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
    printer("using time: %.2f s" % using_time)
    printer('---------------------------')


def compute_MAP_AUC(sim):
    MAP = 0
    AUC = 0
    for i in range(len(sim)):
        ele_key = sim[i].argsort()[::-1]
        for j in range(len(ele_key)):
            if ele_key[j] == i:
                ra = j + 1
                MAP += 1 / ra
                AUC += (sim.shape[1] - ra) / (sim.shape[1] - 1)
                break
    n_nodes = len(sim)
    MAP = MAP / n_nodes
    AUC = AUC / n_nodes
    return MAP, AUC


def get_hits_from_topk(topk):
    count = 0
    for i in range(len(topk)):
        if i in topk[i]:
            count += 1
    return count


def get_topk_index(sim, k):
    ind = np.argpartition(sim, -k)[:, -k:]
    return ind


def get_statistics(alignment_matrix, truth, get_all_metric=False):
    # JUST for crosslin
    source_nodes = list(truth.keys())
    target_nodes = list(truth.values())
    sim = ((alignment_matrix[source_nodes]).T[target_nodes]).T
    if get_all_metric:
        top_k = (1, 10, 50, 100)
    else:
        top_k = [1]
    accs = {}
    acc = None
    for k in top_k:
        topk = get_topk_index(sim, k)
        count = get_hits_from_topk(topk)
        acc = count / len(truth)
        if get_all_metric:
            accs[k] = acc
    if get_all_metric:
        MAP, AUC = compute_MAP_AUC(sim)
        return accs, MAP, AUC
    return acc


# endregion


class MetricCalculator:
    """
        MetricCalculator aims to
        1) address all the statistic tasks.
        2) provide interfaces for querying results.

        MetricCalculator is expected to be used by "evaluation_process".
    """

    def __init__(self,
                 hr_t, tr_h,
                 hits=[1, 3, 5, 10, 50, 100]):

        self.hr_t = hr_t
        self.tr_h = tr_h
        self.hits = hits

        # (f)mr  : (filtered) mean rank
        # (f)mrr : (filtered) mean reciprocal rank
        # (f)hit : (filtered) hit-k ratio
        self.mr = {}
        self.fmr = {}
        self.mrr = {}
        self.fmrr = {}
        self.hit = {}
        self.fhit = {}

        self.rank_head = []
        self.rank_tail = []
        self.f_rank_head = []
        self.f_rank_tail = []
        self.epoch = None
        self.start_time = timeit.default_timer()
        self.reset()

    def reset(self):
        # temporarily used buffers and indexes.
        self.rank_head = []
        self.rank_tail = []
        self.f_rank_head = []
        self.f_rank_tail = []
        self.epoch = None
        self.start_time = timeit.default_timer()

    def append_result(self, result):
        predict_tail = result[0]
        predict_head = result[1]

        h, r, t = result[2], result[3], result[4]

        self.epoch = result[5]

        t_rank, f_t_rank = self.get_tail_rank(predict_tail, h, r, t)
        h_rank, f_h_rank = self.get_head_rank(predict_head, h, r, t)

        self.rank_head.append(h_rank)
        self.rank_tail.append(t_rank)
        self.f_rank_head.append(f_h_rank)
        self.f_rank_tail.append(f_t_rank)

    def get_tail_rank(self, tail_candidate, h, r, t):
        """Function to evaluate the tail rank.

           Args:
               tail_candidate (list): List of the predicted tails for the given head, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id

            Returns:
                Tensors: Returns tail rank and filtered tail rank
        """
        trank = 0
        ftrank = 0

        for j in range(len(tail_candidate)):
            val = tail_candidate[-j - 1]
            if val != t:
                trank += 1
                ftrank += 1
                if val in self.hr_t[(h, r)]:
                    ftrank -= 1
            else:
                break

        return trank, ftrank

    def get_head_rank(self, head_candidate, h, r, t):
        """Function to evaluate the head rank.

           Args:
               head_candidate (list): List of the predicted head for the given tail, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id

            Returns:
                Tensors: Returns head  rank and filetered head rank
        """
        hrank = 0
        fhrank = 0

        for j in range(len(head_candidate)):
            val = head_candidate[-j - 1]
            if val != h:
                hrank += 1
                fhrank += 1
                if val in self.tr_h[(t, r)]:
                    fhrank -= 1
            else:
                break

        return hrank, fhrank

    def settle(self):
        head_ranks = np.asarray(self.rank_head, dtype=np.float32) + 1
        tail_ranks = np.asarray(self.rank_tail, dtype=np.float32) + 1
        head_franks = np.asarray(self.f_rank_head, dtype=np.float32) + 1
        tail_franks = np.asarray(self.f_rank_tail, dtype=np.float32) + 1

        ranks = np.concatenate((head_ranks, tail_ranks))
        franks = np.concatenate((head_franks, tail_franks))

        self.mr[self.epoch] = np.mean(ranks)
        self.mrr[self.epoch] = np.mean(np.reciprocal(ranks))
        self.fmr[self.epoch] = np.mean(franks)
        self.fmrr[self.epoch] = np.mean(np.reciprocal(franks))

        for hit in self.hits:
            self.hit[(self.epoch, hit)] = np.mean(ranks <= hit, dtype=np.float32)
            self.fhit[(self.epoch, hit)] = np.mean(franks <= hit, dtype=np.float32)

    def get_curr_scores(self):
        scores = {
            'mr': self.mr[self.epoch],
            'fmr': self.fmr[self.epoch],
            'mrr': self.mrr[self.epoch],
            'fmrr': self.fmrr[self.epoch]
        }
        return scores

    def display_summary(self):
        """Function to print the test summary."""
        stop_time = timeit.default_timer()
        test_results = [
            "Epoch: %d --- time: %.2f" % (self.epoch, stop_time - self.start_time),
            '--mr,  filtered mr             : %.4f, %.4f' % (self.mr[self.epoch], self.fmr[self.epoch]),
            '--mrr, filtered mrr            : %.4f, %.4f' % (self.mrr[self.epoch], self.fmrr[self.epoch])
        ]
        for hit in self.hits:
            test_results.append('--hits%d                        : %.4f ' % (hit, (self.hit[(self.epoch, hit)])))
            test_results.append('--filtered hits%d               : %.4f ' % (hit, (self.fhit[(self.epoch, hit)])))
        test_results.append("---------------------------------------------------------")
        test_results.append('')
        return test_results


class Evaluator:
    """Class to perform evaluation of the model.

        Args:
            model (object): Model object
            tuning (bool): Flag to denoting tuning if True

        Examples:
            >>> from toolbox.evaluate.Evaluate import Evaluator
            >>> evaluator = Evaluator(model=model, tuning=True)
            >>> evaluator.test_batch(Session(), 0)
            >>> acc = evaluator.output_queue.get()
            >>> evaluator.stop()
    """

    def __init__(self, model,
                 triplets_test, triplets_valid, hr_t, tr_h,
                 entity_count, relation_count,
                 tuning=False,
                 device="cuda",
                 log=print):
        self.model = model
        self.tuning = tuning
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.log = log
        self.triplets_test = triplets_test
        self.triplets_valid = triplets_valid
        self.metric_calculator = MetricCalculator(hr_t, tr_h)

    def test_tail_rank(self, h, r, topk=-1):
        if hasattr(self.model, 'predict_tail_rank'):
            rank = self.model.predict_tail_rank(torch.LongTensor([h]).to(self.device),
                                                torch.LongTensor([r]).to(self.device), topk=topk)
            return rank.squeeze(0)

        h_batch = torch.LongTensor([h]).repeat([self.entity_count]).to(self.device)
        r_batch = torch.LongTensor([r]).repeat([self.entity_count]).to(self.device)
        entity_array = torch.LongTensor(list(range(self.entity_count))).to(self.device)

        preds = self.model.forward(h_batch, r_batch, entity_array)
        _, rank = torch.topk(preds, k=topk)
        return rank

    def test_head_rank(self, r, t, topk=-1):
        if hasattr(self.model, 'predict_head_rank'):
            rank = self.model.predict_head_rank(torch.LongTensor([t]).to(self.device),
                                                torch.LongTensor([r]).to(self.device), topk=topk)
            return rank.squeeze(0)

        entity_array = torch.LongTensor(list(range(self.entity_count))).to(self.device)
        r_batch = torch.LongTensor([r]).repeat([self.entity_count]).to(self.device)
        t_batch = torch.LongTensor([t]).repeat([self.entity_count]).to(self.device)

        preds = self.model.forward(entity_array, r_batch, t_batch)
        _, rank = torch.topk(preds, k=topk)
        return rank

    def test_rel_rank(self, h, t, topk=-1):
        if hasattr(self.model, 'predict_rel_rank'):
            # TODO: This is not implemented for conve, convkb, proje_pointwise, tucker, interacte and hyper
            rank = self.model.predict_rel_rank(h.to(self.device), t.to(self.device), topk=topk)
            return rank.squeeze(0)

        h_batch = torch.LongTensor([h]).repeat([self.relation_count]).to(self.device)
        rel_array = torch.LongTensor(list(range(self.relation_count))).to(self.device)
        t_batch = torch.LongTensor([t]).repeat([self.relation_count]).to(self.device)

        preds = self.model.forward(h_batch, rel_array, t_batch)
        _, rank = torch.topk(preds, k=topk)
        return rank

    def mini_test(self, epoch=None, test_num=0, debug=False):
        if test_num == 0:
            tot_valid_to_test = len(self.triplets_valid)
        else:
            tot_valid_to_test = min(test_num, len(self.triplets_valid))
        if debug:
            tot_valid_to_test = 10

        self.log("Mini-Testing on [%d/%d] Triples in the valid set." % (tot_valid_to_test, len(self.triplets_valid)))
        return self.test(self.triplets_valid, tot_valid_to_test, epoch=epoch)

    def full_test(self, epoch=None, debug=False):
        tot_valid_to_test = len(self.triplets_test)
        if debug:
            tot_valid_to_test = 10

        self.log("Full-Testing on [%d/%d] Triples in the test set." % (tot_valid_to_test, len(self.triplets_test)))
        return self.test(self.triplets_test, tot_valid_to_test, epoch=epoch)

    def test(self, data, num_of_test, epoch=None):
        self.metric_calculator.reset()

        bar = Progbar(num_of_test)
        count = 0
        for i in range(num_of_test):
            h, r, t = data[i].h, data[i].r, data[i].t

            # generate head batch and predict heads.
            h_tensor = torch.LongTensor([h])
            r_tensor = torch.LongTensor([r])
            t_tensor = torch.LongTensor([t])

            hrank = self.test_head_rank(r_tensor, t_tensor, self.entity_count)
            trank = self.test_tail_rank(h_tensor, r_tensor, self.entity_count)

            result_data = [trank.detach().cpu().numpy(), hrank.detach().cpu().numpy(), h, r, t, epoch]

            self.metric_calculator.append_result(result_data)
            count += 1
            bar.update(count, [("setp", count)])

        self.metric_calculator.settle()
        self.metric_calculator.display_summary()
        return self.metric_calculator.get_curr_scores()
