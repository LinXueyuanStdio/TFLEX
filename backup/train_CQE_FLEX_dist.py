"""
@date: 2021/10/26
@description: 分布式训练，单机多卡
"""
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Set

import click
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ComplexQueryData import query_name_dict, QueryStructure, flatten_query, ComplexQueryDatasetCachePath, ComplexQueryData, FB15k_237_BetaE, FB15k_BetaE, NELL_BetaE, all_tasks, name_query_dict
from backup.dataloader import DistributedTestDataset, DistributedTrainDataset
from toolbox.data.dataloader import SingledirectionalOneShotIterator
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds


# 下面 3 行代码解决 tensor 在不同进程中共享问题。
# 共享是通过读写文件的，如果同时打开文件的进程数太多，会崩。
# 这里是把允许的最大进程数设为 4096，比较大了。
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

L = 1


def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * 1 / 2
    return y


class Projection(nn.Module):
    def __init__(self, dim, hidden_dim=1600, num_layers=2):
        super(Projection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, q_feature, q_logic, r_feature, r_logic):
        x = torch.cat([q_feature + r_feature, q_logic + r_logic], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        feature, logic = torch.chunk(x, 2, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        return feature, logic


class Intersection(nn.Module):
    def __init__(self, dim):
        super(Intersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, feature, logic):
        # feature: N x B x d
        # logic:  N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logic = torch.prod(logic, dim=0)
        return feature, logic


class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_feature(self, feature):
        # f,f' in [-L, L]
        # f' = (f + L) % (2L), where L=1
        indicator_positive = feature >= 0
        indicator_negative = feature < 0
        feature[indicator_positive] = feature[indicator_positive] - 1
        feature[indicator_negative] = feature[indicator_negative] + 1
        return feature

    def forward(self, feature, logic):
        feature = self.neg_feature(feature)
        logic = 1 - logic
        return feature, logic


class FLEX(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 center_reg=None, drop: float = 0.):
        super(FLEX, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        # entity only have feature part but no logic part
        self.entity_feature_embedding = nn.Embedding(nentity, self.entity_dim)
        self.relation_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.projection = Projection(self.entity_dim)
        self.intersection = Intersection(self.entity_dim)
        self.negation = Negation()

        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        embedding_range = self.embedding_range.item()
        self.modulus = nn.Parameter(torch.Tensor([0.5 * embedding_range]), requires_grad=True)
        self.cen = center_reg

    def init(self):
        embedding_range = self.embedding_range.item()
        nn.init.uniform_(tensor=self.entity_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)

    def scale(self, embedding):
        return embedding / self.embedding_range

    def forward(self,
                train_data_list: Optional[List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
                test_data_list: Optional[List[Tuple[str, torch.Tensor, torch.Tensor]]]):
        if train_data_list is not None:
            return self.forward_train(train_data_list)
        elif test_data_list is not None:
            return self.forward_test(test_data_list)
        else:
            raise Exception("Train or Test, please choose one!")

    def forward_train(self, data_list: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        positive_scores, negative_scores, subsampling_weights = [], [], []
        for query_name, query, positive_answer, negative_answer, subsampling_weight in data_list:
            positive_score, negative_score = self.forward_predict_2(query_name, query, positive_answer, negative_answer)
            positive_scores.append(positive_score)
            negative_scores.append(negative_score)
            subsampling_weights.append(subsampling_weight)
        positive_scores = torch.cat(positive_scores, dim=0)
        negative_scores = torch.cat(negative_scores, dim=0)
        subsampling_weights = torch.cat(subsampling_weights, dim=0)
        return positive_scores, negative_scores, subsampling_weights

    def forward_test(self, data_list: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        return {"Pe": (B, L) }
        L 是答案个数，[预测实体]和[预测时间戳]的答案个数不一样，所以不能对齐合并
        不同结构的 L 不同
        一般用于valid/test，不用于train
        """
        grouped_score = {}

        for query_name, query, answer in data_list:
            # query (B, L), B for batch size, L for query args length
            # answer (B, N)
            grouped_score[query_name] = self.forward_predict(query_name, query, answer)

        return grouped_score

    def forward_FLEX(self,
                     positive_answer: Optional[torch.Tensor],
                     negative_answer: Optional[torch.Tensor],
                     subsampling_weight: Optional[torch.Tensor],
                     grouped_query: Dict[QueryStructure, torch.Tensor],
                     grouped_idxs: Dict[QueryStructure, List[List[int]]]):
        """
        positive_answer: None or (B, )
        negative_answer: None or (B, N)
        subsampling_weight: None or (B, )
        """
        # 1. 将 查询 嵌入
        all_idxs, all_feature, all_logic = [], [], []
        all_union_idxs, all_union_feature, all_union_logic = [], [], []
        for query_structure in grouped_query:
            # 用字典重新组织了嵌入，一个批次(BxL)只对应一种结构
            if 'u' in query_name_dict[query_structure] and 'DNF' in query_name_dict[query_structure]:
                feature, logic, _ = self.embed_query(self.transform_union_query(grouped_query[query_structure], query_structure),
                                                     self.transform_union_structure(query_structure),
                                                     0)
                all_union_idxs.extend(grouped_idxs[query_structure])
                all_union_feature.append(feature)
                all_union_logic.append(logic)
            else:
                feature, logic, _ = self.embed_query(grouped_query[query_structure],
                                                     query_structure,
                                                     0)
                all_idxs.extend(grouped_idxs[query_structure])
                all_feature.append(feature)
                all_logic.append(logic)

        if len(all_feature) > 0:
            all_feature = torch.cat(all_feature, dim=0).unsqueeze(1)  # (B, 1, d)
            all_logic = torch.cat(all_logic, dim=0).unsqueeze(1)  # (B, 1, d)
        if len(all_union_feature) > 0:
            all_union_feature = torch.cat(all_union_feature, dim=0).unsqueeze(1)  # (2B, 1, d)
            all_union_logic = torch.cat(all_union_logic, dim=0).unsqueeze(1)  # (2B, 1, d)
            all_union_feature = all_union_feature.view(all_union_feature.shape[0] // 2, 2, 1, -1)  # (B, 2, 1, d)
            all_union_logic = all_union_logic.view(all_union_logic.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        # 2. 计算正例损失
        if type(positive_answer) != type(None):
            # 2.1 计算 一般的查询
            if len(all_feature) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_answer[all_idxs]

                positive_feature = self.entity_feature_embedding(positive_sample_regular).unsqueeze(1)
                positive_feature = self.scale(positive_feature)
                positive_feature = convert_to_feature(positive_feature)

                positive_logit = self.scoring_entity(positive_feature, all_feature, all_logic)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)

            # 2.1 计算 并查询
            if len(all_union_feature) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_answer[all_union_idxs]

                positive_feature = self.entity_feature_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_feature = self.scale(positive_feature)
                positive_feature = convert_to_feature(positive_feature)

                positive_union_logit = self.scoring_entity(positive_feature, all_union_feature, all_union_logic)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        # 3. 计算负例损失
        if type(negative_answer) != type(None):
            # 3.1 计算 一般的查询
            if len(all_feature) > 0:
                negative_sample_regular = negative_answer[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape

                negative_feature = self.entity_feature_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_feature = self.scale(negative_feature)
                negative_feature = convert_to_feature(negative_feature)

                negative_logit = self.scoring_entity(negative_feature, all_feature, all_logic)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)

            # 3.1 计算 并查询
            if len(all_union_feature) > 0:
                negative_sample_union = negative_answer[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape

                negative_feature = self.entity_feature_embedding(negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_feature = self.scale(negative_feature)
                negative_feature = convert_to_feature(negative_feature)

                negative_union_logit = self.scoring_entity(negative_feature, all_union_feature, all_union_logic)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def forward_predict(self, query_name: str, query_tensor: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        # query_tensor  # (B, L), B for batch size, L for query args length
        # answer  # (B, N)
        query_structure = name_query_dict[query_name]
        if 'u' in query_name and 'DNF' in query_name:
            feature, logic, _ = self.embed_query(self.transform_union_query(query_tensor, query_structure),
                                                 self.transform_union_structure(query_structure),
                                                 0)
            feature = feature.view(-1, 2, 1, self.entity_dim)
            logic = logic.view(-1, 2, 1, self.entity_dim)
            return self.scoring_to_answers(answer, feature, logic, DNF_predict=True)
        else:
            feature, logic, _ = self.embed_query(query_tensor, query_structure, 0)
            feature = feature.view(-1, 1, 1, self.entity_dim)
            logic = logic.view(-1, 1, 1, self.entity_dim)
            return self.scoring_to_answers(answer, feature, logic, DNF_predict=False)

    def forward_predict_2(self, query_name: str, query_tensor: torch.Tensor,
                          positive_answer: torch.Tensor, negative_answer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query_tensor  # (B, L), B for batch size, L for query args length
        # answer  # (B, N)
        query_structure = name_query_dict[query_name]
        if 'u' in query_name and 'DNF' in query_name:
            feature, logic, _ = self.embed_query(self.transform_union_query(query_tensor, query_structure),
                                                 self.transform_union_structure(query_structure),
                                                 0)
            feature = feature.view(-1, 2, 1, self.entity_dim)
            logic = logic.view(-1, 2, 1, self.entity_dim)
            return self.scoring_to_answers(positive_answer, feature, logic, DNF_predict=True),\
                   self.scoring_to_answers(negative_answer, feature, logic, DNF_predict=True)
        else:
            feature, logic, _ = self.embed_query(query_tensor, query_structure, 0)
            feature = feature.view(-1, 1, 1, self.entity_dim)
            logic = logic.view(-1, 1, 1, self.entity_dim)
            return self.scoring_to_answers(positive_answer, feature, logic, DNF_predict=False),\
                   self.scoring_to_answers(negative_answer, feature, logic, DNF_predict=False)

    def entity_feature(self, idx):
        return convert_to_feature(self.scale(self.entity_feature_embedding(idx)))

    def scoring_to_answers(self, answer_ids: torch.Tensor, feature: torch.Tensor, logic: torch.Tensor, DNF_predict=False):
        """
        B for batch size
        N for negative sampling size (maybe N=1 when positive samples only)
        answer_ids:   (B, N) int
        all_predict:  (B, 1, 1, dt) or (B, 2, 1, dt) float
        return score: (B, N) float
        """
        answer_feature = self.entity_feature(answer_ids).unsqueeze(dim=1)  # (B, 1, N, d)
        scores = self.scoring_entity(answer_feature, feature, logic)  # (B, 1, N) or (B, 2, N)
        if DNF_predict:
            scores = torch.max(scores, dim=1)[0]  # (B, N)
        else:
            scores = scores.squeeze(dim=1)  # (B, N)
        return scores  # (B, N)

    def embed_query(self, queries: torch.Tensor, query_structure, idx: int):
        """
        迭代嵌入
        例子：(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in'
        B = 2, queries=[[1]]
        Iterative embed a batch of queries with same structure using BetaE
        queries:(B, L): a flattened batch of queries (all queries are of query_structure), B is batch size, L is length of queries
        """
        all_relation_flag = True
        for ele in query_structure[-1]:
            # whether the current query tree has merged to one branch and only need to do relation traversal,
            # e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            # 这一类如下
            #     ('e', ('r',)): '1p',
            #     ('e', ('r', 'r')): '2p',
            #     ('e', ('r', 'r', 'r')): '3p',
            #     ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
            #     ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
            # 都是左边是[实体, 中间推理状态]，右边是[关系, 否运算]，只用把状态沿着运算的方向前进一步
            # 所以对 query_structure 的索引只有 0 (左) 和 -1 (右)
            if query_structure[0] == 'e':
                # 嵌入实体
                feature_entity_embedding = self.entity_feature_embedding(queries[:, idx])
                feature_entity_embedding = self.scale(feature_entity_embedding)
                feature_entity_embedding = convert_to_feature(feature_entity_embedding)

                logic_entity_embedding = torch.zeros_like(feature_entity_embedding).to(feature_entity_embedding.device)

                idx += 1

                q_feature = feature_entity_embedding
                q_logic = logic_entity_embedding
            else:
                # 嵌入中间推理状态
                q_feature, q_logic, idx = self.embed_query(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    q_feature, q_logic = self.negation(q_feature, q_logic)

                # projection
                else:
                    r_feature = self.relation_feature_embedding(queries[:, idx])
                    r_feature = self.scale(r_feature)
                    r_feature = convert_to_feature(r_feature)

                    r_logic = self.relation_logic_embedding(queries[:, idx])
                    r_logic = self.scale(r_logic)
                    r_logic = convert_to_feature(r_logic)

                    q_feature, q_logic = self.projection(q_feature, q_logic, r_feature, r_logic)
                idx += 1
        else:
            # 这一类比较复杂，表示最后一个运算是且运算
            #     (('e', ('r',)), ('e', ('r',))): '2i',
            #     (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
            #     (('e', ('r', 'r')), ('e', ('r',))): 'pi',
            #     (('e', ('r',)), ('e', ('r', 'n'))): '2in',
            #     (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
            #     (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
            #     (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
            # intersection
            feature_list = []
            logic_list = []
            for i in range(len(query_structure)):  # 把内部每个子结构都嵌入了，再执行 且运算
                q_feature, q_logic, idx = self.embed_query(queries, query_structure[i], idx)
                feature_list.append(q_feature)
                logic_list.append(q_logic)

            stacked_feature = torch.stack(feature_list)
            stacked_logic = torch.stack(logic_list)

            q_feature, q_logic = self.intersection(stacked_feature, stacked_logic)

        return q_feature, q_logic, idx

    # implement distance function
    def distance(self, entity_feature, query_feature, query_logic):
        d_center = entity_feature - query_feature
        d_left = entity_feature - (query_feature - query_logic)
        d_right = entity_feature - (query_feature + query_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        distance_base = torch.abs(query_logic)
        inner_distance = torch.min(feature_distance, distance_base)

        # outer distance
        indicator_in = feature_distance < distance_base
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right))
        outer_distance[indicator_in] = 0.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance

    def scoring_entity(self, entity_feature, query_feature, query_logic):
        distance = self.distance(entity_feature, query_feature, query_logic)
        score = self.gamma - distance * self.modulus
        return score

    def transform_union_query(self, queries, query_structure: QueryStructure):
        """
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        """
        if query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure: QueryStructure) -> QueryStructure:
        if query_name_dict[query_structure] == '2u-DNF':
            return 'e', ('r',)
        elif query_name_dict[query_structure] == 'up-DNF':
            return 'e', ('r', 'r')


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: ComplexQueryData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, negative_sample_size,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, tasks, evaluate_union, cpu_num,
                 hidden_dim, input_dropout, gamma, center_reg, local_rank
                 ):
        super(MyExperiment, self).__init__(output, local_rank)
        if local_rank == 0:
            self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        nentity = data.nentity
        nrelation = data.nrelation
        nprocs = torch.cuda.device_count()
        self.nprocs = nprocs
        world_size = nprocs  # TODO 暂时用单机多卡。多机多卡需要重新设置
        self.world_size = world_size
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        # batch_size = batch_size // nprocs
        # test_batch_size = test_batch_size // nprocs
        if local_rank == 0:
            self.log('-------------------------------' * 3)
            self.log('# entity: %d' % nentity)
            self.log('# relation: %d' % nrelation)
            self.log('# max steps: %d' % max_steps)
            self.log('Evaluate unoins using: %s' % evaluate_union)
            self.log("loading data")

        # 1. build train dataset
        train_queries = data.train_queries
        train_answers = data.train_answers
        valid_queries = data.valid_queries
        valid_hard_answers = data.valid_hard_answers
        valid_easy_answers = data.valid_easy_answers
        test_queries = data.test_queries
        test_hard_answers = data.test_hard_answers
        test_easy_answers = data.test_easy_answers

        if local_rank == 0:
            self.log("Training info:")
            for query_structure in train_queries:
                self.log(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
            self.log("Validation info:")
            for query_structure in valid_queries:
                self.log(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
            self.log("Test info:")
            for query_structure in test_queries:
                self.log(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))

        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_dataset = DistributedTrainDataset(train_path_queries, nentity, nrelation, negative_sample_size, train_answers)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            train_path_dataset,
            sampler=DistributedSampler(train_path_dataset),
            batch_size=batch_size,
            # shuffle=True,
            num_workers=cpu_num,
            collate_fn=DistributedTrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_dataset = DistributedTrainDataset(train_other_queries, nentity, nrelation, negative_sample_size, train_answers)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                train_other_dataset,
                sampler=DistributedSampler(train_other_dataset),
                batch_size=batch_size,
                # shuffle=True,
                num_workers=cpu_num,
                collate_fn=DistributedTrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

        valid_queries = flatten_query(valid_queries)
        valid_dataset = DistributedTestDataset(valid_queries, nentity, nrelation, valid_easy_answers, valid_hard_answers)
        valid_dataloader = DataLoader(
            valid_dataset,
            sampler=DistributedSampler(valid_dataset),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=DistributedTestDataset.collate_fn
        )

        test_queries = flatten_query(test_queries)
        test_dataset = DistributedTestDataset(test_queries, nentity, nrelation, test_easy_answers, test_hard_answers)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=DistributedSampler(test_dataset),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=DistributedTestDataset.collate_fn
        )

        # 2. build model
        model = FLEX(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=hidden_dim,
            gamma=gamma,
            center_reg=center_reg,
            drop=input_dropout,
        ).cuda(local_rank)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        best_score = 0
        best_test_score = 0
        if resume:
            if resume_by_score > 0:
                start_step, _, best_score = self.model_param_store.load_by_score(model, opt, resume_by_score)
            else:
                start_step, _, best_score = self.model_param_store.load_best(model, opt)
            if local_rank == 0:
                self.dump_model(model)
                model.eval()
                with torch.no_grad():
                    self.debug("Resumed from score %.4f." % best_score)
                    self.debug("Take a look at the performance after resumed.")
                    self.debug("Validation (step: %d):" % start_step)
                    result = self.evaluate(model, valid_dataloader, local_rank)
                    best_score, _ = self.visual_result(start_step + 1, result, "Valid")
                    self.debug("Test (step: %d):" % start_step)
                    result = self.evaluate(model, test_dataloader, local_rank)
                    best_test_score, _ = self.visual_result(start_step + 1, result, "Test")
        else:
            model.init()
            if local_rank == 0:
                self.dump_model(model)
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

        current_learning_rate = lr
        if local_rank == 0:
            hyper = {
                'center_reg': center_reg,
                'tasks': tasks,
                'learning_rate': lr,
                'batch_size': batch_size,
                "hidden_dim": hidden_dim,
                "gamma": gamma,
            }
            self.metric_log_store.add_hyper(hyper)
            for k, v in hyper.items():
                self.log(f'{k} = {v}')
            self.metric_log_store.add_progress(max_steps)
        warm_up_steps = max_steps // 2

        # 3. training
        progbar = Progbar(max_step=max_steps)
        for step in range(start_step, max_steps):
            model.train()
            log = self.train(model, opt, train_path_iterator, step, local_rank)
            for metric in log:
                self.vis.add_scalar('path_' + metric, log[metric], step)
            if train_other_iterator is not None:
                log = self.train(model, opt, train_other_iterator, step, local_rank)
                for metric in log:
                    self.vis.add_scalar('other_' + metric, log[metric], step)
                log = self.train(model, opt, train_path_iterator, step, local_rank)

            if local_rank == 0:
                progbar.update(step + 1, [("step", step + 1), ("loss", log["loss"]), ("positive", log["positive_sample_loss"]), ("negative", log["negative_sample_loss"])])
            if (step + 1) % 10 == 0:
                self.metric_log_store.add_loss(log, step + 1)

            if (step + 1) >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                if local_rank == 0:
                    print("")
                    self.log('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    if local_rank == 0:
                        print("")
                        self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_dataloader, local_rank)
                    if local_rank == 0:
                        score, row_results = self.visual_result(step + 1, result, "Valid")
                        if score >= best_score:
                            self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                            best_score = score
                            self.metric_log_store.add_best_metric({"result": result}, "Valid")
                            self.debug("saving best score %.4f" % score)
                            self.model_param_store.save_best(model, opt, step, 0, score)
                            self.latex_store.save_best_valid_result(row_results)
                        else:
                            self.model_param_store.save_by_score(model, opt, step, 0, score)
                            self.latex_store.save_valid_result_by_score(row_results, score)
                            self.fail("current score=%.4f < best score=%.4f" % (score, best_score))
            if (step + 1) % every_test_step == 0:
                model.eval()
                with torch.no_grad():
                    if local_rank == 0:
                        print("")
                        self.debug("Test (step: %d):" % (step + 1))
                    result = self.evaluate(model, test_dataloader, local_rank)
                    if local_rank == 0:
                        score, row_results = self.visual_result(step + 1, result, "Test")
                        self.latex_store.save_test_result_by_score(row_results, score)
                        if score >= best_test_score:
                            best_test_score = score
                            self.latex_store.save_best_test_result(row_results)
                            self.metric_log_store.add_best_metric({"result": result}, "Test")
                        print("")
        if local_rank == 0:
            self.metric_log_store.finish()

    def train(self, model, optimizer, train_iterator, step, device: Union[str, int] = "cuda:0"):
        model.train()
        model.cuda(device)

        data_list = next(train_iterator)
        cuda_data_list = []
        for query_name, query_tensor, positive_answer, negative_answer, subsampling_weight in data_list:
            query_tensor = query_tensor.cuda(device, non_blocking=True)
            positive_answer = positive_answer.cuda(device, non_blocking=True)
            negative_answer = negative_answer.cuda(device, non_blocking=True)
            subsampling_weight = subsampling_weight.cuda(device, non_blocking=True)
            cuda_data_list.append((query_name, query_tensor, positive_answer, negative_answer, subsampling_weight))

        positive_logit, negative_logit, subsampling_weight = model(cuda_data_list, None)

        negative_sample_loss = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_sample_loss = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_sample_loss).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_sample_loss).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        torch.distributed.barrier()
        log = {
            'positive_sample_loss': reduce_mean(positive_sample_loss, self.nprocs).item(),
            'negative_sample_loss': reduce_mean(negative_sample_loss, self.nprocs).item(),
            'loss': reduce_mean(loss, self.nprocs).item(),
        }

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return log

    def evaluate(self, model, test_dataloader, device: Union[str, int] = "cuda:0"):
        model.cuda(device)
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps)
        logs = defaultdict(list)
        step = 0
        h10 = None
        for data_list, grouped_easy_answer, grouped_hard_answer in test_dataloader:
            cuda_data_list = []
            for query_name, query_tensor, candidate_answer in data_list:
                query_tensor = query_tensor.cuda(device, non_blocking=True)
                candidate_answer = candidate_answer.cuda(device, non_blocking=True)
                cuda_data_list.append((query_name, query_tensor, candidate_answer))

            grouped_score = model(None, cuda_data_list)
            for query_name in grouped_score:
                score = grouped_score[query_name]
                easy_answer_mask: List[torch.Tensor] = grouped_easy_answer[query_name]
                hard_answer: List[Set[int]] = grouped_hard_answer[query_name]
                score[easy_answer_mask] = -float('inf')  # we remove easy answer, because easy answer may exist in training set
                ranking = score.argsort(dim=1, descending=True).cpu()  # sorted idx (B, N)

                ranks = []
                hits = []
                for i in range(10):
                    hits.append([])
                num_queries = ranking.shape[0]
                for i in range(num_queries):
                    for answer_id in hard_answer[i]:
                        rank = torch.where(ranking[i] == answer_id)[0][0]
                        ranks.append(1 / (rank + 1))
                        for hits_level in range(10):
                            hits[hits_level].append(1.0 if rank <= hits_level else 0.0)
                mrr = torch.mean(torch.FloatTensor(ranks)).item()
                h1 = torch.mean(torch.FloatTensor(hits[0])).item()
                h3 = torch.mean(torch.FloatTensor(hits[2])).item()
                h10 = torch.mean(torch.FloatTensor(hits[9])).item()
                logs[query_name].append({
                    'MRR': ranks,
                    'hits@1': hits[0],
                    'hits@2': hits[1],
                    'hits@3': hits[2],
                    'hits@5': hits[4],
                    'hits@10': hits[9],
                    'num_queries': num_queries,
                })

            if device == 0:
                step += 1
                progbar.update(step, [("MRR", mrr), ("Hits @10", h10)])

        # sync and reduce
        # 分布式评估 很麻烦，多任务学习的分布式评估更麻烦，因为每个进程采样到的任务数不一样
        # 下面这坨就是在绝对视野里强行对齐所有进程的任务结果，进行 reduce，最后汇总给 rank=0 的 master node 展示到命令行
        torch.distributed.barrier()  # 所有进程运行到这里时，都等待一下，直到所有进程都在这行，然后一起同时往后分别运行
        query_name_keys = []
        metric_name_keys = []
        all_tensors = []
        metric_names = list(logs[list(logs.keys())[0]][0].keys())
        for query_name in all_tasks:  # test_query_structures 内是所有任务
            for metric_name in metric_names:
                query_name_keys.append(query_name)
                metric_name_keys.append(metric_name)
                if query_name in logs:
                    if metric_name == "num_queries":
                        value = sum([log[metric_name] for log in logs[query_name]])
                        all_tensors.append([value, 1])
                    else:
                        values = []
                        for log in logs[query_name]:
                            values.extend(log[metric_name])
                        all_tensors.append([sum(values), len(values)])
                else:
                    all_tensors.append([0, 1])
        all_tensors = torch.FloatTensor(all_tensors).to(device)
        metrics = defaultdict(lambda: defaultdict(float))
        dist.reduce(all_tensors, dst=0)  # 再次同步，每个进程都分享自己的 all_tensors 给其他进程，每个进程都看到所有数据了
        # 每个进程看到的数据都是所有数据的和，如果有 n 个进程，则有 all_tensors = 0.all_tensors + 1.all_tensors + ... + n.all_tensors
        if dist.get_rank() == 0:  # 我们只在 master 节点上处理，其他进程的结果丢弃了
            # 1. store to dict
            m = defaultdict(lambda: defaultdict(lambda x:[0, 1]))
            for query_name, metric, value in zip(query_name_keys, metric_name_keys, all_tensors):
                m[query_name][metric] = value
            # 2. delete empty query
            del_query = []
            for query_name in m:
                if m[query_name]["num_queries"][0] == 0:
                    del_query.append(query_name)
            for query_name in del_query:
                del m[query_name]
                query_name_keys.remove(query_name)
            # 3. correct values
            for query_name, metric in zip(query_name_keys, metric_name_keys):
                if query_name not in m:
                    continue
                value = m[query_name][metric]
                if metric == "num_queries":
                    metrics[query_name][metric] = int(value[0])
                else:
                    metrics[query_name][metric] = value[0] / value[1]

        return metrics

    def visual_result(self, step_num: int, result, scope: str):
        """Evaluate queries in dataloader"""
        self.metric_log_store.add_metric({scope: result}, step_num, scope)
        average_metrics = defaultdict(float)
        num_query_structures = 0
        num_queries = 0
        for query_structure in result:
            for metric in result[query_structure]:
                self.vis.add_scalar("_".join([scope, query_structure, metric]), result[query_structure][metric], step_num)
                if metric != 'num_queries':
                    average_metrics[metric] += result[query_structure][metric]
            num_queries += result[query_structure]['num_queries']
            num_query_structures += 1

        for metric in average_metrics:
            average_metrics[metric] /= num_query_structures
            self.vis.add_scalar("_".join([scope, 'average', metric]), average_metrics[metric], step_num)

        header = "{0:<8s}".format(scope)
        row_results = defaultdict(list)
        row_results[header].append("avg")
        row_results["num_queries"].append(num_queries)
        for row in average_metrics:
            cell = average_metrics[row]
            row_results[row].append(cell)
        for col in sorted(result):
            row_results[header].append(col)
            col_data = result[col]
            for row in col_data:
                cell = col_data[row]
                row_results[row].append(cell)

        def to_str(data):
            if isinstance(data, float):
                return "{0:>6.2%}  ".format(data)
            elif isinstance(data, int):
                return "{0:^6d}  ".format(data)
            else:
                return "{0:^6s}  ".format(data)

        for i in row_results:
            row = row_results[i]
            self.log("{0:<8s}".format(i)[:8] + ": " + "".join([to_str(data) for data in row]))

        score = average_metrics["MRR"]
        return score, row_results


@click.command()
@click.option("--data_home", type=str, default="data/reasoning", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, NELL.")
@click.option("--name", type=str, default="FLEX_base", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=300001, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10000, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=10000, help="Number of steps.")
@click.option("--batch_size", type=int, default=512, help="Batch size.")
@click.option("--test_batch_size", type=int, default=8, help="Test batch size.")
@click.option('--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.0001, help="Learning rate.")
@click.option('--tasks', type=str, default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
@click.option('--evaluate_union', type=str, default="DNF", help='choices=[DNF, DM] the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
@click.option('--cpu_num', type=int, default=4, help="used to speed up torch.dataloader")
@click.option('--hidden_dim', type=int, default=800, help="embedding dimension")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option('--gamma', type=float, default=30.0, help="margin in the loss")
@click.option('--center_reg', type=float, default=0.02, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
@click.option('--local_rank', type=int, default=-1, help='node rank for distributed training')
def main(data_home, dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, negative_sample_size,
         train_device, test_device,
         resume, resume_by_score,
         lr, tasks, evaluate_union, cpu_num,
         hidden_dim, input_dropout, gamma, center_reg, local_rank
         ):
    set_seeds(0)
    output = OutputSchema(dataset + "-" + name)

    if dataset == "FB15k-237":
        dataset = FB15k_237_BetaE(data_home)
    elif dataset == "FB15k":
        dataset = FB15k_BetaE(data_home)
    elif dataset == "NELL":
        dataset = NELL_BetaE(data_home)
    cache = ComplexQueryDatasetCachePath(dataset.root_path)
    data = ComplexQueryData(cache_path=cache)
    data.load(evaluate_union, tasks)

    MyExperiment(
        output, data,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, negative_sample_size,
        train_device, test_device,
        resume, resume_by_score,
        lr, tasks, evaluate_union, cpu_num,
        hidden_dim, input_dropout, gamma, center_reg, local_rank
    )


if __name__ == '__main__':
    main()
