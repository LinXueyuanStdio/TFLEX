"""
@date: 2021/10/26
@description: null
"""
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set, Union

from toolbox.data.DataSchema import DatasetCachePath
from toolbox.data.DatasetSchema import BaseDatasetSchema
from toolbox.data.functional import read_cache, cache_data

QueryStructure = Union[
    Tuple[str, Tuple[str]],
    Tuple[str, Tuple[str, str]],
    Tuple[str, Tuple[str, str, str]],
    Tuple[Tuple[str, Tuple[str]], Tuple[str, Tuple[str]]],
    Tuple[Tuple[str, Tuple[str]], Tuple[str, Tuple[str]], Tuple[str, Tuple[str]]],
    Tuple[Tuple[Tuple[str, Tuple[str]], Tuple[str, Tuple[str]]], Tuple[str]],
    Tuple[Tuple[str, Tuple[str, str]], Tuple[str, Tuple[str]]],
    Tuple[Tuple[str, Tuple[str]], Tuple[str, Tuple[str, str]]],
    Tuple[Tuple[str, Tuple[str]], Tuple[str, Tuple[str]], Tuple[str, Tuple[str, str]]],
    Tuple[Tuple[Tuple[str, Tuple[str]], Tuple[str, Tuple[str, str]]], Tuple[str]]
]
QueryFlattenIds = Union[
    Tuple[int, Tuple[int]],
    Tuple[int, Tuple[int, int]],
    Tuple[int, Tuple[int, int, int]],
    Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int]]],
    Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int]], Tuple[int, Tuple[int]]],
    Tuple[Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int]]], Tuple[int]],
    Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int]]],
    Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int, int]]],
    Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int]], Tuple[int, Tuple[int, int]]],
    Tuple[Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int, int]]], Tuple[int]]
]
FlattenQueryIdStructure = List[int]

query_name_dict: Dict[QueryStructure, str] = {
    ('e', ('r',)): '1p',
    ('e', ('r', 'r')): '2p',
    ('e', ('r', 'r', 'r')): '3p',
    (('e', ('r',)), ('e', ('r',))): '2i',
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
}
name_query_dict: Dict[str, QueryStructure] = {value: key for key, value in query_name_dict.items()}
all_tasks: List[str] = list(name_query_dict.keys())


# all_tasks = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']

def flatten_query(queries) -> List[Tuple[QueryFlattenIds, QueryStructure]]:
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


class ComplexQueryDatasetCachePath(DatasetCachePath):
    def __init__(self, cache_path: Path):
        DatasetCachePath.__init__(self, cache_path)
        self.train_queries_answers_path = self.cache_path / "train-queries-answers.pkl"
        self.train_queries_path = self.cache_path / "train-queries.pkl"
        self.train_answers_path = self.cache_path / "train-answers.pkl"
        self.valid_queries_path = self.cache_path / "valid-queries.pkl"
        self.valid_hard_answers_path = self.cache_path / "valid-hard-answers.pkl"
        self.valid_easy_answers_path = self.cache_path / "valid-easy-answers.pkl"
        self.test_queries_path = self.cache_path / "test-queries.pkl"
        self.test_hard_answers_path = self.cache_path / "test-hard-answers.pkl"
        self.test_easy_answers_path = self.cache_path / "test-easy-answers.pkl"
        self.stats_path = self.cache_path / "stats.txt"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cache_path})"


class ComplexQueryData:
    def __init__(self, cache_path: ComplexQueryDatasetCachePath):
        self.cache_path = cache_path
        self.train_queries_answers: Dict[QueryStructure, List[Tuple[QueryFlattenIds, Set[int]]]] = {}
        self.train_queries: Dict[QueryStructure, Set[QueryFlattenIds]] = {}
        self.train_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.valid_queries: Dict[QueryStructure, Set[QueryFlattenIds]] = {}
        self.valid_hard_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.valid_easy_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.test_queries: Dict[QueryStructure, Set[QueryFlattenIds]] = {}
        self.test_hard_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.test_easy_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.nentity: int = 0
        self.nrelation: int = 0

    def load_for_scoring_all(self, evaluate_union, tasks):
        if not self.cache_path.train_queries_answers_path.exists():

            def transform_queries_answers(queries: List[Tuple[QueryFlattenIds, QueryStructure]], answers: Dict[QueryFlattenIds, Set[int]]):
                q = defaultdict(list)
                for query, structure in queries:
                    q[structure].append((query, answers[query]))
                return q

            train_queries = read_cache(self.cache_path.train_queries_path)
            train_answers = read_cache(self.cache_path.train_answers_path)
            self.train_queries_answers = transform_queries_answers(flatten_query(train_queries), train_answers)
            cache_data(self.train_queries_answers, self.cache_path.train_queries_answers_path)
        else:
            self.train_queries_answers = read_cache(self.cache_path.train_queries_answers_path)
        self.valid_queries = read_cache(self.cache_path.valid_queries_path)
        self.valid_hard_answers = read_cache(self.cache_path.valid_hard_answers_path)
        self.valid_easy_answers = read_cache(self.cache_path.valid_easy_answers_path)
        self.test_queries = read_cache(self.cache_path.test_queries_path)
        self.test_hard_answers = read_cache(self.cache_path.test_hard_answers_path)
        self.test_easy_answers = read_cache(self.cache_path.test_easy_answers_path)
        with open(str(self.cache_path.stats_path)) as f:
            entrel = f.readlines()
            self.nentity = int(entrel[0].split(' ')[-1])
            self.nrelation = int(entrel[1].split(' ')[-1])

        # remove tasks not in tasks
        for name in all_tasks:
            if 'u' in name:
                name, evaluate_union = name.split('-')
            else:
                evaluate_union = evaluate_union
            if name not in tasks or evaluate_union != evaluate_union:
                query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
                if query_structure in self.train_queries_answers:
                    del self.train_queries_answers[query_structure]
                if query_structure in self.valid_queries:
                    del self.valid_queries[query_structure]
                if query_structure in self.test_queries:
                    del self.test_queries[query_structure]

    def load(self, evaluate_union, tasks):
        self.train_queries = read_cache(self.cache_path.train_queries_path)
        self.train_answers = read_cache(self.cache_path.train_answers_path)
        self.valid_queries = read_cache(self.cache_path.valid_queries_path)
        self.valid_hard_answers = read_cache(self.cache_path.valid_hard_answers_path)
        self.valid_easy_answers = read_cache(self.cache_path.valid_easy_answers_path)
        self.test_queries = read_cache(self.cache_path.test_queries_path)
        self.test_hard_answers = read_cache(self.cache_path.test_hard_answers_path)
        self.test_easy_answers = read_cache(self.cache_path.test_easy_answers_path)
        with open(str(self.cache_path.stats_path)) as f:
            entrel = f.readlines()
            self.nentity = int(entrel[0].split(' ')[-1])
            self.nrelation = int(entrel[1].split(' ')[-1])

        # remove tasks not in tasks
        for name in all_tasks:
            if 'u' in name:
                name, evaluate_union = name.split('-')
            else:
                evaluate_union = evaluate_union
            if name not in tasks or evaluate_union != evaluate_union:
                query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
                if query_structure in self.train_queries:
                    del self.train_queries[query_structure]
                if query_structure in self.valid_queries:
                    del self.valid_queries[query_structure]
                if query_structure in self.test_queries:
                    del self.test_queries[query_structure]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cache_path})"


class FB15k_237_BetaE(BaseDatasetSchema):
    def __init__(self, home: str = "data/reasoning"):
        super(FB15k_237_BetaE, self).__init__("FB15k-237-betae", home)


class FB15k_BetaE(BaseDatasetSchema):
    def __init__(self, home: str = "data/reasoning"):
        super(FB15k_BetaE, self).__init__("FB15k-betae", home)


class NELL_BetaE(BaseDatasetSchema):
    def __init__(self, home: str = "data/reasoning"):
        super(NELL_BetaE, self).__init__("NELL-betae", home)
