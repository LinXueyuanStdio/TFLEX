
from pathlib import Path
from typing import Union, Dict
import click
from random import randint
from ComplexQueryData import *
from ComplexTemporalQueryData import ComplexTemporalQueryDatasetCachePath, TemporalComplexQueryData
import expression
from expression.ParamSchema import get_param_name_list
from toolbox.data.DatasetSchema import RelationalTripletDatasetSchema


class Migration_TFLEX(RelationalTripletDatasetSchema):
    def get_data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.get_dataset_path_child('train'),
            'test': self.get_dataset_path_child('test'),
            'valid': self.get_dataset_path_child('valid'),
        }

    def get_dataset_path(self):
        return self.root_path

class FB15k_237_TFLEX(Migration_TFLEX):
    def __init__(self, home: Union[Path, str] = "data"):
        super(FB15k_237_TFLEX, self).__init__("FB15k_237", home)

class FB15k_TFLEX(Migration_TFLEX):
    def __init__(self, home: Union[Path, str] = "data"):
        super(FB15k_TFLEX, self).__init__("FB15k", home)

class NELL_TFLEX(Migration_TFLEX):
    def __init__(self, home: Union[Path, str] = "data"):
        super(NELL_TFLEX, self).__init__("NELL", home)

flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, tuple) else [l]

@click.command()
@click.option("--data_home", type=str, default="data/reasoning", help="The folder path to source dataset.")
@click.option("--temporal_data_home", type=str, default="data", help="The folder path to dest dataset.")
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, NELL.")
@click.option("--ts", type=int, default=0, help="0 for one fake timestamp, n>0 for up to n fake timestamps.")
def main(data_home, temporal_data_home, dataset, ts):
    print("load static QEs")
    tasks = '1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up'
    evaluate_union = "DNF"
    suffix = "simple" if ts == 0 else f"time_{ts}"
    temporal_data_home = temporal_data_home if ts == 0 else (Path(temporal_data_home) / suffix)
    if dataset == "FB15k-237":
        static_dataset = FB15k_237_BetaE(data_home)
        temporal_dataset = FB15k_237_TFLEX(temporal_data_home)
    elif dataset == "FB15k":
        static_dataset = FB15k_BetaE(data_home)
        temporal_dataset = FB15k_TFLEX(temporal_data_home)
    elif dataset == "NELL":
        static_dataset = NELL_BetaE(data_home)
        temporal_dataset = NELL_TFLEX(temporal_data_home)
    cache = ComplexQueryDatasetCachePath(static_dataset.root_path)
    data = ComplexQueryData(cache_path=cache)
    data.load(evaluate_union, tasks)

    temporal_cache_path = temporal_dataset.cache_path
    temporal_cache = ComplexTemporalQueryDatasetCachePath(temporal_cache_path)

    # begin migration
    query_mapping = {
        "1p": "Pe",
        "2p": "Pe2",
        "3p": "Pe3", # 1p, 2p, 3p
        "2i": "e2i",
        "3i": "e3i", # 2i, 3i

        "pni": "e2i_NPe",
        "pin": "e2i_PeN",
        "inp": "Pe_e2i_Pe_NPe",  # pni, pin, inp
        "2in": "e2i_N",
        "3in": "e3i_N", # 2in, 3in

        "pi": "e2i_Pe",
        "ip": "Pe_e2i",  # pi, ip

        "2u": "e2u",
        "up": "Pe_e2u",  # 2u, up
    }

    # query_structures = {
    #     # 1. 1-hop Pe and Pt, manually
    #     # "Pe": "def Pe(e1, r1, t1): return Pe(e1, r1, t1)",  # 1p
    #     # 2. entity multi-hop
    #     "Pe2": "def Pe2(e1, r1, t1, r2, t2): return Pe(Pe(e1, r1, t1), r2, t2)",  # 2p
    #     "Pe3": "def Pe3(e1, r1, t1, r2, t2, r3, t3): return Pe(Pe(Pe(e1, r1, t1), r2, t2), r3, t3)",  # 3p
    #     # 4. entity and & time and
    #     "e2i": "def e2i(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2i
    #     "e3i": "def e3i(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Pe(e3, r3, t3))",  # 3i
    #     # 5. entity not
    #     "e2i_N": "def e2i_N(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2)))",  # 2in
    #     "e3i_N": "def e3i_N(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Not(Pe(e3, r3, t3)))",  # 3in
    #     "Pe_e2i_Pe_NPe": "def Pe_e2i_Pe_NPe(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2))), r3, t3)",  # inp
    #     "e2i_PeN": "def e2i_PeN(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Not(Pe(e2, r3, t3)))",  # pin
    #     "e2i_NPe": "def e2i_NPe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Not(Pe(Pe(e1, r1, t1), r2, t2)), Pe(e2, r3, t3))",  # pni = e2i_N(Pe(e1, r1, t1), r2, t2, e2, r3, t3)
    #     # 7. entity union & time union
    #     "e2i_Pe": "def e2i_Pe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Pe(e2, r3, t3))",  # pi
    #     "Pe_e2i": "def Pe_e2i(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(e2i(e1, r1, t1, e2, r2, t2), r3, t3)",  # ip
    #     "e2u": "def e2u(e1, r1, t1, e2, r2, t2): return Or(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2u
    #     "Pe_e2u": "def Pe_e2u(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Or(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)",  # up
    # }
    # query_name_dict: Dict[QueryStructure, str] = {
    #     ('e', ('r',)): '1p',
    #     ('e', ('r', 'r')): '2p',
    #     ('e', ('r', 'r', 'r')): '3p',
    #     (('e', ('r',)), ('e', ('r',))): '2i',
    #     (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
    #     ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    #     (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    #     (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    #     (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
    #     ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
    #     (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
    #     (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
    #     (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
    #     ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
    #     ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
    #     ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
    # }
    query_args_mapping = {
        "1p": [0, 1, -1], # ('e', ('r',)) -> Pe(e1, r1, t1)
        "2p": [0, 1, -1, 2, -1],  # ('e', ('r', 'r')) -> Pe(Pe(e1, r1, t1), r2, t2)
        "3p": [0, 1, -1, 2, -1, 3, -1],  # ('e', ('r', 'r', 'r')) -> Pe(Pe(Pe(e1, r1, t1), r2, t2), r3, t3)
        # 1p, 2p, 3p
        "2i": [0, 1, -1, 2, 3, -1],  # (('e', ('r',)), ('e', ('r',))) -> And(Pe(e1, r1, t1), Pe(e2, r2, t2))
        "3i": [0, 1, -1, 2, 3, -1, 4, 5, -1],  # (('e', ('r',)), ('e', ('r',)), ('e', ('r',))) -> And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Pe(e3, r3, t3))
        # 2i, 3i
        "pi": [0, 1, -1, 2, -1, 3, 4, -1],  # (('e', ('r', 'r')), ('e', ('r',))) -> And(Pe(Pe(e1, r1, t1), r2, t2), Pe(e2, r3, t3))
        "ip": [0, 1, -1, 2, 3, -1, 4, -1],  # (('e', ('r',)), ('e', ('r',)), ('e', ('r',))) -> Pe(e2i(e1, r1, t1, e2, r2, t2), r3, t3)
        # pi, ip

        "pni": [0, 1, -1, 2, -1, 4, 5, -1], # (('e', ('r', 'r', 'n')), ('e', ('r',))) -> And(Not(Pe(Pe(e1, r1, t1), r2, t2)), Pe(e2, r3, t3))
        "pin": [0, 1, -1, 2, -1, 3, 4, -1], # (('e', ('r', 'r')), ('e', ('r', 'n'))) -> And(Pe(Pe(e1, r1, t1), r2, t2), Not(Pe(e2, r3, t3)))
        "inp": [0, 1, -1, 2, 3, -1, 5, -1], # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)) -> Pe(And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2))), r3, t3)
        # npi, pni, inp
        "2in": [0, 1, -1, 2, 3, -1], # (('e', ('r',)), ('e', ('r', 'n'))) -> And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2)))
        "3in": [0, 1, -1, 2, 3, -1, 4, 5, -1], # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))) -> And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Not(Pe(e3, r3, t3)))
        # 2in, 3in

        "2u": [0, 1, -1, 2, 3, -1], # (('e', ('r',)), ('e', ('r',)), ('u',)) -> Or(Pe(e1, r1, t1), Pe(e2, r2, t2))
        "up": [0, 1, -1, 2, 3, -1, 5, -1], # ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)) -> Pe(Or(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)
        # 2u, up
    }
    def EntityProjection(s, r, t): return {}
    def TimeProjection(s, r, o): return {}
    def TimeBefore(t): return {}
    def TimeAfter(t): return {}
    def TimeNext(t): return {}
    parser = expression.BasicParser({}, {
        "EntityProjection": EntityProjection,
        "TimeProjection": TimeProjection,
        "TimeBefore": TimeBefore,
        "TimeAfter": TimeAfter,
        "TimeNext": TimeNext,
    })
    # train
    print("migrate train TQEs")
    split = "train"
    train_queries_answers = {}
    for name in query_mapping:
        query_name = query_mapping[name]
        query_structure_name = query_name

        query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
        if query_structure not in data.train_queries:
            print("skip", name)
            continue
        queries: Set[QueryFlattenIds] = data.train_queries[query_structure]

        qas = []
        args_mapping = query_args_mapping[name]
        print("migrating", name)
        for ids in queries:
            answers = data.train_answers[ids]
            ids = flatten(ids)
            new_ids = list(range(len(args_mapping)))
            for idx, arg_idx in enumerate(args_mapping):
                if arg_idx == -1:
                    new_ids[idx] = randint(0, ts)
                else:
                    new_ids[idx] = ids[arg_idx]
            qas.append((new_ids, answers))

        param_name_list = get_param_name_list(parser.eval(query_structure_name))
        train_queries_answers[query_structure_name] = {
            "args": param_name_list,
            "queries_answers": qas
        }

        path = temporal_cache.cache_queries_answers_path(split, query_name)
        cache_data(train_queries_answers[query_structure_name], path)
        cache_data(train_queries_answers, temporal_cache.cache_train_queries_answers_path)

    # valid
    print("migrate valid TQEs")
    split = "valid"
    valid_queries_answers = {}
    for name in query_mapping:
        query_name = query_mapping[name]
        query_structure_name = query_name

        query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
        if query_structure not in data.valid_queries:
            print("skip", name)
            continue
        queries: Set[QueryFlattenIds] = data.valid_queries[query_structure]

        qas = []
        args_mapping = query_args_mapping[name]
        print("migrating", name)
        for ids in queries:
            hard_answers = data.valid_hard_answers[ids]
            easy_answers = data.valid_easy_answers[ids]
            total_answers = hard_answers.union(easy_answers)
            ids = flatten(ids)
            new_ids = list(range(len(args_mapping)))
            for idx, arg_idx in enumerate(args_mapping):
                if arg_idx == -1:
                    new_ids[idx] = randint(0, ts)
                else:
                    new_ids[idx] = ids[arg_idx]
            qas.append((new_ids, easy_answers, total_answers))

        param_name_list = get_param_name_list(parser.eval(query_structure_name))
        valid_queries_answers[query_structure_name] = {
            "args": param_name_list,
            "queries_answers": qas
        }
        cache_data(valid_queries_answers[query_structure_name], temporal_cache.cache_queries_answers_path(split, query_name))
        cache_data(valid_queries_answers, temporal_cache.cache_valid_queries_answers_path)

    # test
    print("migrate test TQEs")
    split = "test"
    test_queries_answers = {}
    for name in query_mapping:
        query_name = query_mapping[name]
        query_structure_name = query_name

        query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
        if query_structure not in data.test_queries:
            print("skip", name)
            continue
        queries: Set[QueryFlattenIds] = data.test_queries[query_structure]

        qas = []
        args_mapping = query_args_mapping[name]
        print("migrating", name)
        for ids in queries:
            hard_answers = data.test_hard_answers[ids]
            easy_answers = data.test_easy_answers[ids]
            total_answers = hard_answers.union(easy_answers)
            ids = flatten(ids)
            new_ids = list(range(len(args_mapping)))
            for idx, arg_idx in enumerate(args_mapping):
                if arg_idx == -1:
                    new_ids[idx] = randint(0, ts)
                else:
                    new_ids[idx] = ids[arg_idx]
            qas.append((new_ids, easy_answers, total_answers))

        param_name_list = get_param_name_list(parser.eval(query_structure_name))
        test_queries_answers[query_structure_name] = {
            "args": param_name_list,
            "queries_answers": qas
        }
        cache_data(test_queries_answers[query_structure_name], temporal_cache.cache_queries_answers_path(split, query_name))
        cache_data(test_queries_answers, temporal_cache.cache_test_queries_answers_path)

    # meta
    print("migrate meta")

    query_meta = {}

    def avg_answers_count(qa):
        return sum([len(row[-1]) for row in qa]) / len(qa) if len(qa) > 0 else 0

    for query_name in test_queries_answers.keys():
        train_qa = train_queries_answers[query_name]["queries_answers"] if query_name in train_queries_answers else []
        valid_qa = valid_queries_answers[query_name]["queries_answers"] if query_name in valid_queries_answers else []
        test_qa = test_queries_answers[query_name]["queries_answers"] if query_name in test_queries_answers else []
        queries_answers = train_qa + valid_qa + test_qa
        query_meta[query_name] = {
            "queries_count": len(queries_answers),
            "avg_answers_count": avg_answers_count(queries_answers),
            "train": {
                "queries_count": len(train_qa),
                "avg_answers_count": avg_answers_count(train_qa),
            },
            "valid": {
                "queries_count": len(valid_qa),
                "avg_answers_count": avg_answers_count(valid_qa),
            },
            "test": {
                "queries_count": len(test_qa),
                "avg_answers_count": avg_answers_count(test_qa),
            },
        }
    meta = {
        "entity_count": data.nentity,
        "relation_count": data.nrelation,
        "timestamp_count": 1 if ts == 0 else ts,
        "query_meta": query_meta,
        # ignore below
        "valid_triples_count": -1,
        "test_triples_count": -1,
        "train_triples_count": -1,
        "triple_count": -1,
    }
    cache_data(meta, temporal_cache.cache_metadata_path)

    # end migration
    print("done")
    # have a look
    print("unit testing")
    data = TemporalComplexQueryData(temporal_dataset, cache_path=temporal_cache)
    data.preprocess_data_if_needed()
    data.load_cache([
        "meta",
    ])

    entity_count = data.entity_count
    relation_count = data.relation_count
    timestamp_count = data.timestamp_count
    train_tasks = "Pe,Pe2,Pe3,e2i,e3i,e2i_N,e3i_N,Pe_e2i_Pe_NPe,e2i_PeN,e2i_NPe"
    tasks = ["Pe", "Pe2", "Pe3", "e2i", "e3i", "e2i_N", "e3i_N", "Pe_e2i_Pe_NPe", "e2i_PeN", "e2i_NPe", "e2u", "Pe_e2u"]
    data.train_queries_answers = data.load_cache_by_tasks(train_tasks.split(","), "train")
    data.valid_queries_answers = data.load_cache_by_tasks(tasks, "valid")
    data.test_queries_answers = data.load_cache_by_tasks(tasks, "test")
    print("passed")


if __name__ == '__main__':
    main()