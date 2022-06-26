"""
@date: 2021/10/26
@description: null
"""
import logging
import os
import os.path as osp
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple

import click
import numpy as np

from toolbox.data.functional import read_triple_ids_htr2hrt


def set_logger(save_path, query_name, print_on_screen=False):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(save_path, '%s.log' % (query_name))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def index_dataset(dataset_name, force=False):
    print('Indexing dataset {0}'.format(dataset_name))
    base_path = 'data/{0}/'.format(dataset_name)
    # files = ['train.txt', 'valid.txt', 'test.txt']
    # indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']
    files = ['train.txt']
    indexified_files = ['train_indexified.txt']
    return_flag = True
    for i in range(len(indexified_files)):
        if not osp.exists(osp.join(base_path, indexified_files[i])):
            return_flag = False
            break
    if return_flag and not force:
        print("index file exists")
        return

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}

    entid, relid = 0, 0

    with open(osp.join(base_path, files[0])) as f:
        lines = f.readlines()
        file_len = len(lines)

    for p, indexified_p in zip(files, indexified_files):
        fw = open(osp.join(base_path, indexified_p), "w")
        with open(osp.join(base_path, p), 'r') as f:
            for i, line in enumerate(f):
                print('[%d/%d]' % (i, file_len), end='\r')
                e1, rel, e2 = line.split('\t')
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = '-' + rel
                rel = '+' + rel
                # rel_reverse = rel+ '_reverse'

                if p == "train.txt":
                    if e1 not in ent2id.keys():
                        ent2id[e1] = entid
                        id2ent[entid] = e1
                        entid += 1

                    if e2 not in ent2id.keys():
                        ent2id[e2] = entid
                        id2ent[entid] = e2
                        entid += 1

                    if not rel in rel2id.keys():
                        rel2id[rel] = relid
                        id2rel[relid] = rel
                        assert relid % 2 == 0
                        relid += 1

                    if not rel_reverse in rel2id.keys():
                        rel2id[rel_reverse] = relid
                        id2rel[relid] = rel_reverse
                        assert relid % 2 == 1
                        relid += 1

                if e1 in ent2id.keys() and e2 in ent2id.keys():
                    fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                    fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
        fw.close()

    with open(osp.join(base_path, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, 'ent2id.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'rel2id.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('num entity: %d, num relation: %d' % (len(ent2id), len(rel2id)))
    print("indexing finished!!")


def construct_graph_from_triples(triples: List[Tuple[int, int, int]]):
    # knowledge graph
    # triples [(h_idx, r_idx, t_idx)]
    # kb[e][rel] = set([e, e, e])
    ent_in = defaultdict(lambda: defaultdict(set))  # [t][r] -> {h_idx} ent 作为 t，入链
    ent_out = defaultdict(lambda: defaultdict(set))  # [h][r] -> {t_idx} ent 作为 h，出链
    for e1, rel, e2 in triples:
        ent_in[e2][rel].add(e1)
        ent_out[e1][rel].add(e2)
    return ent_in, ent_out


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def write_links(dataset, ent_out, small_ent_out, max_ans_num, name):
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    num_more_answer = 0
    for ent in ent_out:
        for rel in ent_out[ent]:
            if len(ent_out[ent][rel]) <= max_ans_num:
                queries[('e', ('r',))].add((ent, (rel,)))
                tp_answers[(ent, (rel,))] = small_ent_out[ent][rel]
                fn_answers[(ent, (rel,))] = ent_out[ent][rel]
            else:
                num_more_answer += 1

    with open('./data/%s/%s-queries.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(tp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fp_answers, f)
    print(num_more_answer)


def ground_queries(dataset, query_structure, ent_in, ent_out, small_ent_in, small_ent_out, gen_num, max_ans_num, query_name, mode):
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
    tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    s0 = time.time()
    old_num_sampled = -1
    while num_sampled < gen_num:
        if num_sampled != 0 and num_sampled % (gen_num // 100) == 0 and num_sampled != old_num_sampled:
            logging.info(f'{mode} {query_structure}: [{num_sampled:d}/{gen_num:d}], avg time: {(time.time() - s0) / num_sampled}, try: {num_try}, repeat: {num_repeat}: more_answer: {num_more_answer}, broken: {num_broken}, no extra: {num_no_extra_answer}, no negative: {num_no_extra_negative} empty: {num_empty}')
            old_num_sampled = num_sampled
        print(
            f'{mode} {query_structure}: [{num_sampled:d}/{gen_num:d}], avg time: {(time.time() - s0) / (num_sampled + 0.001)}, try: {num_try}, repeat: {num_repeat}: more_answer: {num_more_answer}, broken: {num_broken}, no extra: {num_no_extra_answer}, no negative: {num_no_extra_negative} empty: {num_empty}', end='\r')
        num_try += 1
        empty_query_structure = deepcopy(query_structure)
        answer = random.sample(ent_in.keys(), 1)[0]
        broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer)
        if broken_flag:
            num_broken += 1
            continue
        query = empty_query_structure
        answer_set = achieve_answer(query, ent_in, ent_out)
        small_answer_set = achieve_answer(query, small_ent_in, small_ent_out)
        if len(answer_set) == 0:
            num_empty += 1
            continue
        if mode != 'train':
            if len(answer_set - small_answer_set) == 0:
                num_no_extra_answer += 1
                continue
            if 'n' in query_name:
                if len(small_answer_set - answer_set) == 0:
                    num_no_extra_negative += 1
                    continue
        if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > max_ans_num:
            num_more_answer += 1
            continue
        if list2tuple(query) in queries[list2tuple(query_structure)]:
            num_repeat += 1
            continue
        queries[list2tuple(query_structure)].add(list2tuple(query))
        tp_answers[list2tuple(query)] = small_answer_set
        fp_answers[list2tuple(query)] = small_answer_set - answer_set
        fn_answers[list2tuple(query)] = answer_set - small_answer_set
        num_sampled += 1
        tp_ans_num.append(len(tp_answers[list2tuple(query)]))
        fp_ans_num.append(len(fp_answers[list2tuple(query)]))
        fn_ans_num.append(len(fn_answers[list2tuple(query)]))

    print()
    logging.info("{} tp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(tp_ans_num), np.min(tp_ans_num), np.mean(tp_ans_num), np.std(tp_ans_num)))
    logging.info("{} fp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fp_ans_num), np.min(fp_ans_num), np.mean(fp_ans_num), np.std(fp_ans_num)))
    logging.info("{} fn max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fn_ans_num), np.min(fn_ans_num), np.mean(fn_ans_num), np.std(fn_ans_num)))

    name_to_save = '%s-%s' % (mode, query_name)
    with open('./data/%s/%s-queries.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(fp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(tp_answers, f)
    return queries, tp_answers, fp_answers, fn_answers


def generate_queries(dataset, query_structure, query_name,
                     gen_num: Tuple[int, int, int],
                     max_ans_num,
                     gen_train, gen_valid, gen_test,
                     train_triples_ids: List[Tuple[int, int, int]],
                     valid_triples_ids: List[Tuple[int, int, int]],
                     test_triples_ids: List[Tuple[int, int, int]]
                     ):
    print('structure is', query_structure, "with name", query_name)
    set_logger(f"./data/{dataset}/", query_name)
    if gen_train and gen_valid and gen_test:
        # 针对同时生成 3 种数据集的情况，特别优化
        generate_trian_valid_test_queries_by_structure(dataset, query_structure, query_name, gen_num, max_ans_num,
                                                       train_triples_ids, valid_triples_ids, test_triples_ids)
        return

    if gen_train:
        empty_ent_in, empty_ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
        train_ent_in, train_ent_out = construct_graph_from_triples(train_triples_ids)  # ent_in
        generate_queries_by_structure(dataset, query_structure,
                                      train_ent_in, train_ent_out, empty_ent_in, empty_ent_out, train_ent_out,
                                      gen_num[0], max_ans_num, query_name, 'train')
    if gen_valid:
        train_ent_in, train_ent_out = construct_graph_from_triples(train_triples_ids)  # ent_in
        valid_ent_in, valid_ent_out = construct_graph_from_triples(train_triples_ids + valid_triples_ids)
        _, valid_only_ent_out = construct_graph_from_triples(valid_triples_ids)
        generate_queries_by_structure(dataset, query_structure,
                                      valid_ent_in, valid_ent_out, train_ent_in, train_ent_out, valid_only_ent_out,
                                      gen_num[1], max_ans_num, query_name, 'valid')
    if gen_test:
        valid_ent_in, valid_ent_out = construct_graph_from_triples(train_triples_ids + valid_triples_ids)
        test_ent_in, test_ent_out = construct_graph_from_triples(train_triples_ids + valid_triples_ids + test_triples_ids)
        _, test_only_ent_out = construct_graph_from_triples(test_triples_ids)
        generate_queries_by_structure(dataset, query_structure,
                                      test_ent_in, test_ent_out, valid_ent_in, valid_ent_out, test_only_ent_out,
                                      gen_num[2], max_ans_num, query_name, 'test')


def generate_trian_valid_test_queries_by_structure(
        dataset, query_structure, query_name, gen_num, max_ans_num,
        train_triples_ids: List[Tuple[int, int, int]],
        valid_triples_ids: List[Tuple[int, int, int]],
        test_triples_ids: List[Tuple[int, int, int]]):
    empty_ent_in, empty_ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    train_ent_in, train_ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    valid_ent_in, valid_ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    test_ent_in, test_ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    valid_only_ent_out = defaultdict(lambda: defaultdict(set))
    test_only_ent_out = defaultdict(lambda: defaultdict(set))
    for h, r, t in train_triples_ids:
        train_ent_in[t][r].add(h)
        train_ent_out[h][r].add(t)
        valid_ent_in[t][r].add(h)
        valid_ent_out[h][r].add(t)
        test_ent_in[t][r].add(h)
        test_ent_out[h][r].add(t)
    for h, r, t in valid_triples_ids:
        valid_ent_in[t][r].add(h)
        valid_ent_out[h][r].add(t)
        valid_only_ent_out[h][r].add(t)
        test_ent_in[t][r].add(h)
        test_ent_out[h][r].add(t)
    for h, r, t in test_triples_ids:
        test_ent_in[t][r].add(h)
        test_ent_out[h][r].add(t)
        test_only_ent_out[h][r].add(t)
    generate_queries_by_structure(dataset, query_structure, query_name, 'train', gen_num[0], max_ans_num,
                                  train_ent_in, train_ent_out, empty_ent_in, empty_ent_out, train_ent_out)
    generate_queries_by_structure(dataset, query_structure, query_name, 'valid', gen_num[1], max_ans_num,
                                  valid_ent_in, valid_ent_out, train_ent_in, train_ent_out, valid_only_ent_out)
    generate_queries_by_structure(dataset, query_structure, query_name, 'test', gen_num[2], max_ans_num,
                                  test_ent_in, test_ent_out, valid_ent_in, valid_ent_out, test_only_ent_out)


def generate_queries_by_structure(dataset, query_structure, query_name, mode, gen_num, max_ans_num,
                                  ent_in, ent_out, small_ent_in, small_ent_out, only_ent_out):
    if query_structure == ['e', ['r']]:
        write_links(dataset, only_ent_out, small_ent_out, max_ans_num, f"{mode}-{query_name}")
    else:
        ground_queries(dataset, query_structure,
                       ent_in, ent_out, small_ent_in, small_ent_out,
                       gen_num[1], max_ans_num, query_name, mode)
        print('%s queries generated with structure %s' % (gen_num, query_structure))


def fill_query(query_structure, ent_in, ent_out, answer):
    assert type(query_structure[-1]) == list
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == 'n':
                query_structure[-1][i] = -2
                continue
            found = False
            for j in range(40):
                r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                if r_tmp // 2 != r // 2 or r_tmp == r:
                    r = r_tmp
                    found = True
                    break
            if not found:
                return True
            query_structure[-1][i] = r
            answer = random.sample(ent_in[answer][r], 1)[0]
        if query_structure[0] == 'e':
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer)
    else:
        same_structure = defaultdict(list)
        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer)
            if broken_flag:
                return True
        for structure in same_structure:
            if len(same_structure[structure]) != 1:
                structure_set = set()
                for i in same_structure[structure]:
                    structure_set.add(list2tuple(query_structure[i]))
                if len(structure_set) < len(same_structure[structure]):
                    return True


def achieve_answer(query, ent_in, ent_out) -> set:
    assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = {query[0]}
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                # negation
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                # projection
                ent_set_traverse = set()
                for ent in ent_set:
                    ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                ent_set = ent_set_traverse
    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out)
        union_flag = False
        if len(query[-1]) == 1 and query[-1][0] == -1:
            union_flag = True
        for i in range(1, len(query)):
            if union_flag:
                # union
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out))
            else:
                # intersection
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out))
    return ent_set


e = 'e'
r = 'r'
n = 'n'
u = 'u'
query_structures = [
    [e, [r]],
    [e, [r, r]],
    [e, [r, r, r]],
    [[e, [r]], [e, [r]]],
    [[e, [r]], [e, [r]], [e, [r]]],
    [[e, [r, r]], [e, [r]]],
    [[[e, [r]], [e, [r]]], [r]],
    # negation
    [[e, [r]], [e, [r, n]]],
    [[e, [r]], [e, [r]], [e, [r, n]]],
    [[e, [r, r]], [e, [r, n]]],
    [[e, [r, r, n]], [e, [r]]],
    [[[e, [r]], [e, [r, n]]], [r]],
    # union
    [[e, [r]], [e, [r]], [u]],
    [[[e, [r]], [e, [r]], [u]], [r]]
]
query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']
query_structure_name_pairs = [(query_structure, query_name) for query_structure, query_name in zip(query_structures, query_names)]


@click.group("main")
def main():
    print("main group")


@main.command("index", help="建立实体索引和关系索引")
@click.option('--dataset', default="FB15k-237")
@click.option('--reindex', is_flag=True, default=False)
def index(dataset, reindex):
    index_dataset(dataset, reindex)


@click.command("generate_by_dataset", help="按预定义参数生成 query 数据")
@click.option('--dataset', default="FB15k-237")
@click.option('--max_ans_num', default=1e6)
@click.option('--gen_train', is_flag=True, default=False)
@click.option('--gen_valid', is_flag=True, default=False)
@click.option('--gen_test', is_flag=True, default=False)
@click.option('--gen_id', default=0)
def generate_by_dataset(dataset, max_ans_num, gen_train, gen_valid, gen_test, gen_id):
    train_num_dict = {'FB15k': 273710, "FB15k-237": 149689, "NELL": 107982}
    valid_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    test_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    gen_train_num = train_num_dict[dataset]
    gen_valid_num = valid_num_dict[dataset]
    gen_test_num = test_num_dict[dataset]
    generate(dataset, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, gen_train, gen_valid, gen_test, gen_id)


@click.command("generate", help="生成 query 数据")
@click.option('--dataset', default="FB15k-237")
@click.option('--gen_train_num', default=0)
@click.option('--gen_valid_num', default=0)
@click.option('--gen_test_num', default=0)
@click.option('--max_ans_num', default=1e6)
@click.option('--gen_train', is_flag=True, default=False)
@click.option('--gen_valid', is_flag=True, default=False)
@click.option('--gen_test', is_flag=True, default=False)
@click.option('--gen_id', default=0)
def generate(dataset, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, gen_train, gen_valid, gen_test, gen_id):
    query_structure, query_name = query_structure_name_pairs[gen_id]
    # indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']

    train_triples_ids: List[Tuple[int, int, int]] = read_triple_ids_htr2hrt(f"data/{dataset}/train.txt")  # [h_idx, r_id, t_idx], r contains reverse
    valid_triples_ids: List[Tuple[int, int, int]] = read_triple_ids_htr2hrt(f"data/{dataset}/valid.txt")  # [h_idx, r_id, t_idx], r contains reverse
    test_triples_ids: List[Tuple[int, int, int]] = read_triple_ids_htr2hrt(f"data/{dataset}/test.txt")  # [h_idx, r_id, t_idx], r contains reverse

    generate_queries(dataset, query_structure, query_name,
                     (gen_train_num, gen_valid_num, gen_test_num),
                     max_ans_num,
                     gen_train, gen_valid, gen_test,
                     train_triples_ids, valid_triples_ids, test_triples_ids
                     )


cli = click.CommandCollection(sources=[main])

if __name__ == '__main__':
    cli()
