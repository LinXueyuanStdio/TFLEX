import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Union, Set, Dict

import torch


def cache_data(data, cache_path: Union[str, Path]):
    with open(str(cache_path), 'wb') as f:
        pickle.dump(data, f)


def read_cache(cache_path: Union[str, Path]):
    with open(str(cache_path), 'rb') as f:
        return pickle.load(f)


def read_ids_and_names(file_path: Union[str, Path], sp="\t") -> Tuple[List[int], List[str]]:
    ids = []
    names = []
    with open(str(file_path), 'r', encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            id_to_name = line.strip().split(sp)
            ids.append(int(id_to_name[0]))
            names.append(id_to_name[1])
    return ids, names


def read_triple_hrt(file_path: Union[str, Path]) -> List[Tuple[str, str, str]]:
    with open(str(file_path), 'r', encoding='utf-8') as fr:
        triple = set()
        for line in fr:
            line_split = line.split()
            head = line_split[0]
            rel = line_split[1]
            tail = line_split[2]
            triple.add((head, rel, tail))
    return list(triple)


def read_attribute_triple_eav(file_path: Union[str, Path]) -> List[Tuple[str, str, str]]:
    with open(str(file_path), 'r', encoding='utf-8') as fr:
        triple = set()
        for line in fr:
            line_split = line.split()
            entity = line_split[0][1: -1]
            attr = line_split[1][1: -1]
            value = line_split[2]
            triple.add((entity, attr, value))
    return list(triple)


def save_triple_hrt(triples: List[Tuple[str, str, str]], file_path: Union[str, Path]):
    with open(str(file_path), 'w') as fr:
        for triple in triples:
            fr.write("%s\t%s\t%s\n" % (triple[0], triple[1], triple[2]))


def read_triple_ids_hrt(file_path: Union[str, Path]) -> List[Tuple[int, int, int]]:
    with open(str(file_path), 'r', encoding='utf-8') as fr:
        triple = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            rel = int(line_split[1])
            tail = int(line_split[2])
            triple.add((head, rel, tail))
    return list(triple)


def read_triple_ids_htr2hrt(file_path: Union[str, Path]) -> List[Tuple[int, int, int]]:
    with open(str(file_path), 'r') as fr:
        triple = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[1])
            rel = int(line_split[2])
            triple.add((head, rel, tail))
    return list(triple)


def save_triple_ids_hrt(triples: List[Tuple[int, int, int]], file_path: Union[str, Path]):
    with open(str(file_path), 'w') as fr:
        for triple in triples:
            fr.write("%d\t%d\t%d\n" % (triple[0], triple[1], triple[2]))


def append_align_triple(triple: List[Tuple[int, int, int]], entity_align_list: List[Tuple[int, int]]):
    # 使用对齐实体替换头节点，构造属性三元组数据，从而达到利用对齐实体数据的目的
    align_set = {}
    for i in entity_align_list:
        align_set[i[0]] = i[1]
        align_set[i[1]] = i[0]
    triple_replace_with_align = []
    # bar = Progbar(max_step=len(triple))
    # i = 0
    for entity, attr, value in triple:
        if entity in align_set:
            triple_replace_with_align.append((align_set[entity], attr, value))
        if value in align_set:
            triple_replace_with_align.append((entity, attr, align_set[value]))
        if (entity in align_set) and (value in align_set):
            triple_replace_with_align.append((align_set[entity], attr, align_set[value]))
        # i += 1
        # bar.update(i, [("step", i)])
    return triple + triple_replace_with_align


def build_map_tr_h(triplets: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], Set[int]]:
    """ Function to read the list of heads for the given tail and relation pair. """
    tr_h: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for h, r, t in triplets:
        tr_h[(t, r)].add(h)

    return tr_h


def build_map_hr_t(triplets: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], Set[int]]:
    """ Function to read the list of tails for the given head and relation pair. """
    hr_t: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for h, r, t in triplets:
        hr_t[(h, r)].add(t)

    return hr_t


def read_seeds_ids(file_path: Union[str, Path]) -> List[Tuple[int, int]]:
    with open(str(file_path), 'r', encoding='utf-8') as f:
        ret = []
        for line in f:
            th = line[:-1].split('\t')
            ret.append((int(th[0]), int(th[1])))
        return ret


def read_seeds(file_path: Union[str, Path]) -> List[Tuple[str, str]]:
    with open(str(file_path), 'r', encoding='utf-8') as f:
        ret = []
        for line in f:
            th = line[:-1].split('\t')
            ret.append((th[0], th[1]))
        return ret


def edge_idx_and_rel_idx(triples: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.LongTensor(triples)
    edge_index = t[:, [0, 2]].T
    rel = t[:, 1]
    return edge_index, rel


def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    rel_all = torch.cat([rel, rel + rel.max() + 1])
    return edge_index_all, rel_all


def with_inverse_relations(triples_ids: List[Tuple[int, int, int]], max_relation_id: int) -> Tuple[List[Tuple[int, int, int]], torch.Tensor, torch.Tensor]:
    """
    triples_ids:Tx3
    triples:2Tx3
    edge_index_all:2x2T
    rel_all:2T
    """
    edge_index, rel = edge_idx_and_rel_idx(triples_ids)
    edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    rel_all = torch.cat([rel, rel + max_relation_id])
    triples = torch.cat([edge_index_all.T, rel_all.unsqueeze(dim=-1)], dim=1)[:, [0, 2, 1]]
    triples = triples.tolist()
    return triples, edge_index_all, rel_all
