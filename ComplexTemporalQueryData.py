"""
@date: 2022/3/2
@description: null
"""
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import List, Tuple, Dict, Set, Union, Any

import expression
from expression.ParamSchema import placeholder2sample, get_param_name_list, get_placeholder_list, placeholder2fixed, FixedQuery, clear_placeholder_list
from toolbox.data.DataSchema import DatasetCachePath, BaseData
from toolbox.data.DatasetSchema import RelationalTripletDatasetSchema
from toolbox.data.functional import read_cache, cache_data
from toolbox.utils.Progbar import Progbar


class ICEWS14(RelationalTripletDatasetSchema):
    def __init__(self, home: Union[Path, str] = "data"):
        super(ICEWS14, self).__init__("ICEWS14", home)

    def get_data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.get_dataset_path_child('train'),
            'test': self.get_dataset_path_child('test'),
            'valid': self.get_dataset_path_child('valid'),
        }

    def get_dataset_path(self):
        return self.root_path


class ICEWS05_15(RelationalTripletDatasetSchema):
    def __init__(self, home: Union[Path, str] = "data"):
        super(ICEWS05_15, self).__init__("ICEWS05-15", home)

    def get_data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.get_dataset_path_child('train'),
            'test': self.get_dataset_path_child('test'),
            'valid': self.get_dataset_path_child('valid'),
        }

    def get_dataset_path(self):
        return self.root_path


class GDELT(RelationalTripletDatasetSchema):
    def __init__(self, home: Union[Path, str] = "data"):
        super(GDELT, self).__init__("GDELT", home)

    def get_data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.get_dataset_path_child('train'),
            'test': self.get_dataset_path_child('test'),
            'valid': self.get_dataset_path_child('valid'),
        }

    def get_dataset_path(self):
        return self.root_path


class TemporalKnowledgeDatasetCachePath(DatasetCachePath):
    def __init__(self, cache_path: Path):
        DatasetCachePath.__init__(self, cache_path)
        self.cache_all_triples_path = self.cache_path / 'triplets_all.pkl'
        self.cache_train_triples_path = self.cache_path / 'triplets_train.pkl'
        self.cache_test_triples_path = self.cache_path / 'triplets_test.pkl'
        self.cache_valid_triples_path = self.cache_path / 'triplets_valid.pkl'

        self.cache_all_triples_ids_path = self.cache_path / 'triplets_ids_all.pkl'
        self.cache_train_triples_ids_path = self.cache_path / 'triplets_ids_train.pkl'
        self.cache_test_triples_ids_path = self.cache_path / 'triplets_ids_test.pkl'
        self.cache_valid_triples_ids_path = self.cache_path / 'triplets_ids_valid.pkl'

        self.cache_all_entities_path = self.cache_path / 'entities.pkl'
        self.cache_all_relations_path = self.cache_path / 'relations.pkl'
        self.cache_all_timestamps_path = self.cache_path / 'timestamps.pkl'
        self.cache_entities_ids_path = self.cache_path / "entities_ids.pkl"
        self.cache_relations_ids_path = self.cache_path / "relations_ids.pkl"
        self.cache_timestamps_ids_path = self.cache_path / "timestamps_ids.pkl"

        self.cache_idx2entity_path = self.cache_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.cache_path / 'idx2relation.pkl'
        self.cache_idx2timestamp_path = self.cache_path / 'idx2timestamp.pkl'
        self.cache_entity2idx_path = self.cache_path / 'entity2idx.pkl'
        self.cache_relation2idx_path = self.cache_path / 'relation2idx.pkl'
        self.cache_timestamps2idx_path = self.cache_path / 'timestamp2idx.pkl'


def read_triple_srot(file_path: Union[str, Path]) -> List[Tuple[str, str, str, str]]:
    """
    return [(lhs, rel, rhs, timestamp)]
              s    r    o       t
    """
    with open(str(file_path), 'r', encoding='utf-8') as fr:
        triple = set()
        for line in fr.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            triple.add((lhs, rel, rhs, timestamp))
    return list(triple)


TYPE_MAPPING_sro_t = Dict[int, Dict[int, Dict[int, Set[int]]]]
TYPE_MAPPING_srt_o = Dict[int, Dict[int, Dict[int, Set[int]]]]
TYPE_MAPPING_t_sro = Dict[int, Set[Tuple[int, int, int]]]
TYPE_MAPPING_o_srt = Dict[int, Set[Tuple[int, int, int]]]


def build_map_t2sro_and_o2srt(triples_ids: List[Tuple[int, int, int, int]]) -> Tuple[TYPE_MAPPING_t_sro, TYPE_MAPPING_o_srt]:
    t_sro = defaultdict(set)
    o_srt = defaultdict(set)
    for s, r, o, t in triples_ids:
        t_sro[t].add((s, r, o))
        o_srt[o].add((s, r, t))
    return t_sro, o_srt


def build_map_sro_t(triplets: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], Set[int]]:
    """ Function to read the list of tails for the given head and relation pair. """
    sro_t: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
    for s, r, o, t in triplets:
        sro_t[(s, r, o)].add(t)

    return sro_t


def build_map_srt_o(triplets: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], Set[int]]:
    """ Function to read the list of tails for the given head and relation pair. """
    srt_o: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
    for s, r, o, t in triplets:
        srt_o[(s, r, t)].add(o)

    return srt_o


def build_map_sro2t_and_srt2o(triples_ids: List[Tuple[int, int, int, int]]) -> Tuple[TYPE_MAPPING_sro_t, TYPE_MAPPING_srt_o]:
    """ Function to read the list of tails for the given head and relation pair. """
    sro_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    srt_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for s, r, o, t in triples_ids:
        sro_t[s][r][o].add(t)
        srt_o[s][r][t].add(o)
    return sro_t, srt_o


def build_mapping(triples_ids: List[Tuple[int, int, int, int]]):
    sro_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sor_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    srt_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    str_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sot_r = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sto_r = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    ors_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    osr_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ort_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    otr_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ost_r = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ots_r = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    trs_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tsr_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tro_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tor_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tso_r = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tos_r = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    rts_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    rst_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    rto_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    rot_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    rso_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ros_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    t_sro = defaultdict(set)
    o_srt = defaultdict(set)
    s_rot = defaultdict(set)
    r_sot = defaultdict(set)
    for s, r, o, t in triples_ids:
        sro_t[s][r][o].add(t)
        sor_t[s][o][r].add(t)
        srt_o[s][r][t].add(o)
        str_o[s][t][r].add(o)
        sot_r[s][o][t].add(r)
        sto_r[s][t][o].add(r)
        ors_t[o][r][s].add(t)
        osr_t[o][s][r].add(t)
        ort_s[o][r][t].add(s)
        otr_s[o][t][r].add(s)
        ost_r[o][s][t].add(r)
        ots_r[o][t][s].add(r)
        trs_o[t][r][s].add(o)
        tsr_o[t][s][r].add(o)
        tro_s[t][r][o].add(s)
        tor_s[t][o][r].add(s)
        tso_r[t][s][o].add(r)
        tos_r[t][o][s].add(r)
        rts_o[r][t][s].add(o)
        rst_o[r][s][t].add(o)
        rto_s[r][t][o].add(s)
        rot_s[r][o][t].add(s)
        rso_t[r][s][o].add(t)
        ros_t[r][o][s].add(t)
        t_sro[t].add((s, r, o))
        o_srt[o].add((s, r, t))
        s_rot[s].add((r, t, o))
        r_sot[r].add((s, o, t))
    return sro_t, sor_t, srt_o, str_o, sot_r, sto_r, \
           ors_t, osr_t, ort_s, otr_s, ost_r, ots_r, \
           trs_o, tsr_o, tro_s, tor_s, tso_r, tos_r, \
           rts_o, rst_o, rto_s, rot_s, rso_t, ros_t, \
           t_sro, o_srt, s_rot, r_sot


def build_mapping_simple(triples_ids: List[Tuple[int, int, int, int]]):
    sro_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sor_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    srt_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    str_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ors_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    trs_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tsr_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    tro_s = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    rst_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    rso_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    t_sro = defaultdict(set)
    o_srt = defaultdict(set)
    for s, r, o, t in triples_ids:
        sro_t[s][r][o].add(t)
        sor_t[s][o][r].add(t)
        srt_o[s][r][t].add(o)
        str_o[s][t][r].add(o)
        ors_t[o][r][s].add(t)
        trs_o[t][r][s].add(o)
        tsr_o[t][s][r].add(o)
        tro_s[t][r][o].add(s)
        rst_o[r][s][t].add(o)
        rso_t[r][s][o].add(t)
        t_sro[t].add((s, r, o))
        o_srt[o].add((s, r, t))
    return sro_t, sor_t, srt_o, str_o, \
           ors_t, trs_o, tro_s, rst_o, \
           rso_t, t_sro, o_srt


def build_not_t2sro_o2srt(entities_ids: List[int], timestamps_ids: List[int],
                          sro_t: TYPE_MAPPING_sro_t, srt_o: TYPE_MAPPING_srt_o) -> Tuple[TYPE_MAPPING_t_sro, TYPE_MAPPING_o_srt]:
    # DON'T USE THIS FUNCTION! THERE ARE DRAGONS!
    not_t_sro = defaultdict(set)
    not_o_srt = defaultdict(set)
    for s in sro_t:
        for r in sro_t[s]:
            for o in sro_t[s][r]:
                for t in set(timestamps_ids) - set(sro_t[s][r][o]):  # negation on timestamps
                    not_t_sro[t].add((s, r, o))
    for s in srt_o:
        for r in srt_o[s]:
            for t in srt_o[s][r]:
                for o in set(entities_ids) - set(srt_o[s][r][t]):  # negation on entities
                    not_o_srt[o].add((s, r, t))
    return not_t_sro, not_o_srt


class TemporalKnowledgeData(BaseData):
    """ The class is the main module that handles the knowledge graph.

        KnowledgeGraph is responsible for downloading, parsing, processing and preparing
        the training, testing and validation dataset.

        Args:
            dataset (RelationalTripletDatasetSchema): custom dataset.
            cache_path (TemporalKnowledgeDatasetCachePath): cache path.

        Attributes:
            dataset (RelationalTripletDatasetSchema): custom dataset.
            cache_path (TemporalKnowledgeDatasetCachePath): cache path.

            all_relations (list):list of all the relations.
            all_entities (list): List of all the entities.
            all_timestamps (list): List of all the timestamps.

            entity2idx (dict): Dictionary for mapping string name of entities to unique numerical id.
            idx2entity (dict): Dictionary for mapping the entity id to string.
            relation2idx (dict): Dictionary for mapping string name of relations to unique numerical id.
            idx2relation (dict): Dictionary for mapping the relation id to string.
            timestamp2idx (dict): Dictionary for mapping string name of timestamps to unique numerical id.
            idx2timestamp (dict): Dictionary for mapping the timestamp id to string.

        Examples:
            >>> from ComplexTemporalQueryData import ICEWS14, TemporalKnowledgeDatasetCachePath, TemporalKnowledgeData
            >>> dataset = ICEWS14()
            >>> cache = TemporalKnowledgeDatasetCachePath(dataset.cache_path)
            >>> data = TemporalKnowledgeData(dataset=dataset, cache_path=cache)
            >>> data.preprocess_data_if_needed()

    """

    def __init__(self,
                 dataset: RelationalTripletDatasetSchema,
                 cache_path: TemporalKnowledgeDatasetCachePath):
        BaseData.__init__(self, dataset, cache_path)
        self.dataset = dataset
        self.cache_path = cache_path

        # KG data structure stored in triplet format
        self.all_triples: List[Tuple[str, str, str, str]] = []
        self.train_triples: List[Tuple[str, str, str, str]] = []
        self.test_triples: List[Tuple[str, str, str, str]] = []
        self.valid_triples: List[Tuple[str, str, str, str]] = []

        self.all_triples_ids: List[Tuple[int, int, int, int]] = []
        self.train_triples_ids: List[Tuple[int, int, int, int]] = []
        self.test_triples_ids: List[Tuple[int, int, int, int]] = []
        self.valid_triples_ids: List[Tuple[int, int, int, int]] = []

        self.all_relations: List[str] = []
        self.all_entities: List[str] = []
        self.all_timestamps: List[str] = []
        self.entities_ids: List[int] = []
        self.relations_ids: List[int] = []
        self.timestamps_ids: List[int] = []

        self.entity2idx: Dict[str, int] = {}
        self.idx2entity: Dict[int, str] = {}
        self.relation2idx: Dict[str, int] = {}
        self.idx2relation: Dict[int, str] = {}
        self.timestamp2idx: Dict[str, int] = {}
        self.idx2timestamp: Dict[int, str] = {}

        # meta
        self.entity_count = 0
        self.relation_count = 0
        self.timestamp_count = 0
        self.valid_triples_count = 0
        self.test_triples_count = 0
        self.train_triples_count = 0
        self.triple_count = 0

    def read_all_origin_data(self):
        self.read_all_triplets()

    def read_all_triplets(self):
        self.train_triples = read_triple_srot(self.dataset.data_paths['train'])
        self.valid_triples = read_triple_srot(self.dataset.data_paths['valid'])
        self.test_triples = read_triple_srot(self.dataset.data_paths['test'])
        self.all_triples = self.train_triples + self.valid_triples + self.test_triples

        self.valid_triples_count = len(self.valid_triples)
        self.test_triples_count = len(self.test_triples)
        self.train_triples_count = len(self.train_triples)
        self.triple_count = self.valid_triples_count + self.test_triples_count + self.train_triples_count

    def transform_all_data(self):
        self.transform_entities_relations_timestamps()
        self.transform_mappings()
        self.transform_all_triplets_ids()

        self.transform_entity_ids()
        self.transform_relation_ids()
        self.transform_timestamp_ids()

    def transform_entities_relations_timestamps(self):
        """ Function to read the entities. """
        entities: Set[str] = set()
        relations: Set[str] = set()
        timestamps: Set[str] = set()
        # print("entities_relations")
        # bar = Progbar(len(self.all_triples))
        # i = 0
        for s, r, o, t in self.all_triples:
            entities.add(s)
            relations.add(r)
            entities.add(o)
            timestamps.add(t)
            # i += 1
            # bar.update(i, [("h", h.split("/")[-1]), ("r", r.split("/")[-1]), ("t", t.split("/")[-1])])

        self.all_entities = sorted(list(entities))
        self.all_relations = sorted(list(relations))
        self.all_timestamps = sorted(list(timestamps))

        self.entity_count = len(self.all_entities)
        self.relation_count = len(self.all_relations)
        self.timestamp_count = len(self.all_timestamps)

    def transform_mappings(self):
        """ Function to generate the mapping from string name to integer ids. """
        for k, v in enumerate(self.all_entities):
            self.entity2idx[v] = k
            self.idx2entity[k] = v
        for k, v in enumerate(self.all_relations):
            self.relation2idx[v] = k
            self.idx2relation[k] = v
        for k, v in enumerate(self.all_timestamps):
            self.timestamp2idx[v] = k
            self.idx2timestamp[k] = v

    def transform_all_triplets_ids(self):
        entity2idx = self.entity2idx
        relation2idx = self.relation2idx
        timestamp2idx = self.timestamp2idx
        self.train_triples_ids = [(entity2idx[s], relation2idx[r], entity2idx[o], timestamp2idx[t]) for s, r, o, t in self.train_triples]
        self.test_triples_ids = [(entity2idx[s], relation2idx[r], entity2idx[o], timestamp2idx[t]) for s, r, o, t in self.test_triples]
        self.valid_triples_ids = [(entity2idx[s], relation2idx[r], entity2idx[o], timestamp2idx[t]) for s, r, o, t in self.valid_triples]
        self.all_triples_ids = self.train_triples_ids + self.valid_triples_ids + self.test_triples_ids

    def transform_entity_ids(self):
        entity2idx = self.entity2idx
        for e in self.all_entities:
            self.entities_ids.append(entity2idx[e])
        print("entities_ids", len(self.entities_ids))

    def transform_relation_ids(self):
        relation2idx = self.relation2idx
        for r in self.all_relations:
            self.relations_ids.append(relation2idx[r])
        print("relations_ids", len(self.relations_ids))

    def transform_timestamp_ids(self):
        timestamp2idx = self.timestamp2idx
        for t in self.all_timestamps:
            self.timestamps_ids.append(timestamp2idx[t])
        print("timestamps_ids", len(self.timestamps_ids))

    def cache_all_data(self):
        """Function to cache the prepared dataset in the memory"""
        cache_data(self.all_triples, self.cache_path.cache_all_triples_path)
        cache_data(self.train_triples, self.cache_path.cache_train_triples_path)
        cache_data(self.test_triples, self.cache_path.cache_test_triples_path)
        cache_data(self.valid_triples, self.cache_path.cache_valid_triples_path)

        cache_data(self.all_triples_ids, self.cache_path.cache_all_triples_ids_path)
        cache_data(self.train_triples_ids, self.cache_path.cache_train_triples_ids_path)
        cache_data(self.test_triples_ids, self.cache_path.cache_test_triples_ids_path)
        cache_data(self.valid_triples_ids, self.cache_path.cache_valid_triples_ids_path)

        cache_data(self.all_entities, self.cache_path.cache_all_entities_path)
        cache_data(self.all_relations, self.cache_path.cache_all_relations_path)
        cache_data(self.all_timestamps, self.cache_path.cache_all_timestamps_path)
        cache_data(self.entities_ids, self.cache_path.cache_entities_ids_path)
        cache_data(self.relations_ids, self.cache_path.cache_relations_ids_path)
        cache_data(self.timestamps_ids, self.cache_path.cache_timestamps_ids_path)

        cache_data(self.idx2entity, self.cache_path.cache_idx2entity_path)
        cache_data(self.idx2relation, self.cache_path.cache_idx2relation_path)
        cache_data(self.idx2timestamp, self.cache_path.cache_idx2timestamp_path)
        cache_data(self.relation2idx, self.cache_path.cache_relation2idx_path)
        cache_data(self.entity2idx, self.cache_path.cache_entity2idx_path)
        cache_data(self.timestamp2idx, self.cache_path.cache_timestamps2idx_path)

        cache_data(self.meta(), self.cache_path.cache_metadata_path)

    def load_cache(self, keys: List[str]):
        for key in keys:
            self.read_cache_data(key)

    def read_cache_data(self, key):
        """Function to read the cached dataset from the memory"""
        path = "cache_%s_path" % key
        if hasattr(self, key) and hasattr(self.cache_path, path):
            key_path = getattr(self.cache_path, path)
            value = read_cache(key_path)
            setattr(self, key, value)
            return value
        elif key == "meta":
            meta = read_cache(self.cache_path.cache_metadata_path)
            self.read_meta(meta)
        else:
            raise ValueError('Unknown cache data key %s' % key)

    def read_meta(self, meta):
        self.entity_count = meta["entity_count"]
        self.relation_count = meta["relation_count"]
        self.timestamp_count = meta["timestamp_count"]
        self.valid_triples_count = meta["valid_triples_count"]
        self.test_triples_count = meta["test_triples_count"]
        self.train_triples_count = meta["train_triples_count"]
        self.triple_count = meta["triple_count"]

    def meta(self) -> Dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "timestamp_count": self.timestamp_count,
            "valid_triples_count": self.valid_triples_count,
            "test_triples_count": self.test_triples_count,
            "train_triples_count": self.train_triples_count,
            "triple_count": self.triple_count,
        }

    def dump(self) -> List[str]:
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = [
            "",
            "-" * 15 + "Metadata Info for Dataset: " + self.dataset.name + "-" * (15 - len(self.dataset.name)),
            "Total Training Triples   :%s" % self.train_triples_count,
            "Total Testing Triples    :%s" % self.test_triples_count,
            "Total validation Triples :%s" % self.valid_triples_count,
            "Total Entities           :%s" % self.entity_count,
            "Total Relations          :%s" % self.relation_count,
            "Total Timestamps         :%s" % self.timestamp_count,
            "-" * (30 + len("Metadata Info for Dataset: ")),
            "",
        ]
        return dump


"""
above is simple temporal kg
below is complex query data (logical reasoning) based on previous temporal kg
"""


class ComplexTemporalQueryDatasetCachePath(TemporalKnowledgeDatasetCachePath):
    def __init__(self, cache_path: Path):
        TemporalKnowledgeDatasetCachePath.__init__(self, cache_path)
        self.cache_train_queries_answers_path = self.cache_path / "train_queries_answers.pkl"
        self.cache_valid_queries_answers_path = self.cache_path / "valid_queries_answers.pkl"
        self.cache_test_queries_answers_path = self.cache_path / "test_queries_answers.pkl"


TYPE_train_queries_answers = Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]]
TYPE_test_queries_answers = Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int], Set[int]]]]]]


class ComplexQueryData(TemporalKnowledgeData):

    def __init__(self,
                 dataset: RelationalTripletDatasetSchema,
                 cache_path: ComplexTemporalQueryDatasetCachePath):
        TemporalKnowledgeData.__init__(self, dataset, cache_path)
        self.cache_path = cache_path
        # Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]]
        #       |                       |                     |          |
        #     structure name      args name list              |          |
        #                                    ids corresponding to args   |
        #                                                          answers id set
        # 1. `structure name` is the name of a function (named query function), parsed to AST and eval to get results.
        # 2. `args name list` is the arg list of query function.
        # 3. valid_queries_answers and test_queries_answers are the same type as train_queries_answers
        self.train_queries_answers: TYPE_train_queries_answers = {
            # "Pe_aPt": {
            #     "args": ["e1", "r1", "e2", "r2", "e3"],
            #     "queries_answers": [
            #         ([1, 2, 3, 4, 5], {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3, 5}),
            #     ]
            # }
            # answers = Pe_aPt(1, 2, 3, 4, 5)
            # then, answers == {2, 3}
        }
        self.valid_queries_answers: TYPE_test_queries_answers = {
            # "Pe_aPt": {
            #     "args": ["e1", "r1", "e2", "r2", "e3"],
            #     "queries_answers": [
            #         ([1, 2, 3, 4, 5], {2, 3}, {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3}, {2, 3, 5}),
            #         ([1, 2, 3, 4, 5], {2, 3}, {2, 3, 5}),
            #     ]
            # }
            # answers = Pe_aPt(1, 2, 3, 4, 5)
            # in training set, answers == {2, 3}
            # in validation set, answers == {2, 3, 5}, more harder and complete
        }
        self.test_queries_answers: TYPE_test_queries_answers = {}
        # meta
        self.query_meta = {
            # "Pe_aPt": {
            #     "queries_count": 1,
            #     "avg_answers_count": 1
            # }
        }

    def transform_all_data(self):
        TemporalKnowledgeData.transform_all_data(self)
        # 0. prepare data.
        # add inverse relations
        max_relation_id = self.relation_count
        relations_ids_with_reverse = self.relations_ids + [r + max_relation_id for r in self.relations_ids]

        def append_reverse(triples):
            nonlocal max_relation_id
            res = []
            for s, r, o, t in triples:
                res.append((s, r, o, t))
                res.append((o, r + max_relation_id, s, t))
            return res

        train_triples_ids = append_reverse(self.train_triples_ids)
        valid_triples_ids = append_reverse(self.valid_triples_ids)
        test_triples_ids = append_reverse(self.test_triples_ids)

        # 1. 1-hop: Pe, Pt
        train_sro_t, train_srt_o = build_map_sro2t_and_srt2o(self.train_triples_ids)
        valid_sro_t, valid_srt_o = build_map_sro2t_and_srt2o(self.valid_triples_ids)
        test_sro_t, test_srt_o = build_map_sro2t_and_srt2o(self.test_triples_ids)

        def build_one_hop(param_name_list: List[str], sro_t, for_test=False):
            queries_answers = []
            for s in sro_t:
                for r in sro_t[s]:
                    for o in sro_t[s][r]:
                        answers = sro_t[s][r][o]
                        if len(answers) > 0:
                            queries = [s, r, o]
                            if for_test:
                                queries_answers.append((queries, {}, answers))
                            else:
                                queries_answers.append((queries, answers))
            return {
                "args": param_name_list,
                "queries_answers": queries_answers
            }

        if self.cache_path.cache_train_queries_answers_path.exists():
            self.train_queries_answers = read_cache(self.cache_path.cache_train_queries_answers_path)
            self.valid_queries_answers = read_cache(self.cache_path.cache_valid_queries_answers_path)
            self.test_queries_answers = read_cache(self.cache_path.cache_test_queries_answers_path)

        def cache_step():
            cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)
            cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
            cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)

        if "Pe" not in self.train_queries_answers:
            self.train_queries_answers["Pe"] = build_one_hop(["e1", "r1", "t1"], train_srt_o, for_test=False)
            cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)
        if "Pe" not in self.valid_queries_answers:
            self.valid_queries_answers["Pe"] = build_one_hop(["e1", "r1", "t1"], valid_srt_o, for_test=True)
            cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
        if "Pe" not in self.test_queries_answers:
            self.test_queries_answers["Pe"] = build_one_hop(["e1", "r1", "t1"], test_srt_o, for_test=True)
            cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)
        print("Pe",
              "train", len(self.train_queries_answers["Pe"]["queries_answers"]),
              "valid", len(self.valid_queries_answers["Pe"]["queries_answers"]),
              "test", len(self.test_queries_answers["Pe"]["queries_answers"]),
              )

        if "Pt" not in self.train_queries_answers:
            self.train_queries_answers["Pt"] = build_one_hop(["e1", "r1", "e2"], train_sro_t, for_test=False)
            cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)
        if "Pt" not in self.valid_queries_answers:
            self.valid_queries_answers["Pt"] = build_one_hop(["e1", "r1", "e2"], valid_sro_t, for_test=True)
            cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
        if "Pt" not in self.test_queries_answers:
            self.test_queries_answers["Pt"] = build_one_hop(["e1", "r1", "e2"], test_sro_t, for_test=True)
            cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)
        print("Pt",
              "train", len(self.train_queries_answers["Pt"]["queries_answers"]),
              "valid", len(self.valid_queries_answers["Pt"]["queries_answers"]),
              "test", len(self.test_queries_answers["Pt"]["queries_answers"]),
              )

        # 2. multi-hop: Pe_aPt, Pe_bPt, etc
        train_sro_t, train_sor_t, train_srt_o, train_str_o, \
        train_ors_t, train_trs_o, train_tro_s, train_rst_o, \
        train_rso_t, train_t_sro, train_o_srt = build_mapping_simple(train_triples_ids)
        valid_sro_t, valid_sor_t, valid_srt_o, valid_str_o, \
        valid_ors_t, valid_trs_o, valid_tro_s, valid_rst_o, \
        valid_rso_t, valid_t_sro, valid_o_srt = build_mapping_simple(train_triples_ids + valid_triples_ids)
        test_sro_t, test_sor_t, test_srt_o, test_str_o, \
        test_ors_t, test_trs_o, test_tro_s, test_rst_o, \
        test_rso_t, test_t_sro, test_o_srt = build_mapping_simple(train_triples_ids + valid_triples_ids + test_triples_ids)
        # 2.1 parser
        train_parser = expression.SamplingParser(self.entities_ids, relations_ids_with_reverse, self.timestamps_ids,
                                                 train_sro_t, train_sor_t, train_srt_o, train_str_o,
                                                 train_ors_t, train_trs_o, train_tro_s, train_rst_o,
                                                 train_rso_t, train_t_sro, train_o_srt)
        valid_parser = expression.SamplingParser(self.entities_ids, relations_ids_with_reverse, self.timestamps_ids,
                                                 valid_sro_t, valid_sor_t, valid_srt_o, valid_str_o,
                                                 valid_ors_t, valid_trs_o, valid_tro_s, valid_rst_o,
                                                 valid_rso_t, valid_t_sro, valid_o_srt)
        test_parser = expression.SamplingParser(self.entities_ids, relations_ids_with_reverse, self.timestamps_ids,
                                                test_sro_t, test_sor_t, test_srt_o, test_str_o,
                                                test_ors_t, test_trs_o, test_tro_s, test_rst_o,
                                                test_rso_t, test_t_sro, test_o_srt)

        # 2.2. sampling
        # we generate 1p, t-1p according to original train/valid/test triples.
        # for union-DM, we don't need to actually generate it.
        # The model should use 2u, up, t-2u, t-up with DM by itself.
        query_structure_name_list = [
            # entity
            "Pe2", "Pe3", "e2i", "e3i",  # 2p, 3p, 2i, 3i
            "e2i_NPe", "e2i_PeN", "Pe_e2i_Pe_NPe", "e2i_N", "e3i_N",  # npi, pni, inp, 2in, 3in
            # time
            "Pt_lPe", "Pt_rPe", "Pe_Pt", "Pe_aPt", "Pe_bPt", "Pe_nPt",  # t-1p, t-2p
            "t2i", "t3i", "Pt_le2i", "Pt_re2i", "Pe_at2i", "Pe_bt2i", "Pe_nt2i", "between",  # t-2i, t-3i
            "t2i_NPt", "t2i_PtN", "Pe_t2i_PtPe_NPt", "t2i_N", "t3i_N",  # t-npi, t-pni, t-inp, t-2in, t-3in
            # entity
            "e2i_Pe", "Pe_e2i",  # pi, ip
            "e2u", "Pe_e2u",  # 2u, up
            # time
            "t2i_Pe", "Pe_t2i",  # t-pi, t-ip
            "t2u", "Pe_t2u",  # t-2u, t-up
        ]
        # how many samples should we generate?
        max_sample_count = len(build_map_srt_o(train_triples_ids))
        train_sample_counts = {
            # entity
            "Pe2": max_sample_count,
            "Pe3": max_sample_count,
            "e2i": max_sample_count,
            "e3i": max_sample_count,  # 2p, 3p, 2i, 3i
            "e2i_NPe": max_sample_count // 10,
            "e2i_PeN": max_sample_count // 10,
            "Pe_e2i_Pe_NPe": max_sample_count // 10,
            "e2i_N": max_sample_count // 10,
            "e3i_N": max_sample_count // 10,  # npi, pni, inp, 2in, 3in
            # time
            "Pt_lPe": max_sample_count // 10,
            "Pt_rPe": max_sample_count // 10,
            "Pe_Pt": max_sample_count // 10,
            "Pe_aPt": max_sample_count // 10,
            "Pe_bPt": max_sample_count // 10,
            # "Pe_nPt": max_sample_count // 10,  # t-1p, t-2p
            "t2i": max_sample_count,
            "t3i": max_sample_count,
            "Pt_le2i": max_sample_count // 10,
            "Pt_re2i": max_sample_count // 10,
            "Pe_at2i": max_sample_count // 10,
            "Pe_bt2i": max_sample_count // 10,
            # "Pe_nt2i": max_sample_count // 10,
            "between": max_sample_count // 10,  # t-2i, t-3i
            "t2i_NPt": max_sample_count // 10,
            "t2i_PtN": max_sample_count // 10,
            "Pe_t2i_PtPe_NPt": max_sample_count // 10,
            "t2i_N": max_sample_count // 10,
            "t3i_N": max_sample_count // 10,  # t-npi, t-pni, t-inp, t-2in, t-3in
        }
        test_sample_count = min(max_sample_count // 30, 10000)
        test_sample_counts = {
            # entity
            "Pe2": test_sample_count,
            "Pe3": test_sample_count,
            "e2i": test_sample_count,
            "e3i": test_sample_count,  # 2p, 3p, 2i, 3i
            "e2i_NPe": test_sample_count,
            "e2i_PeN": test_sample_count,
            "Pe_e2i_Pe_NPe": test_sample_count,
            "e2i_N": test_sample_count,
            "e3i_N": test_sample_count,  # npi, pni, inp, 2in, 3in
            # time
            "Pt_lPe": test_sample_count,
            "Pt_rPe": test_sample_count,
            "Pe_Pt": test_sample_count,
            "Pe_aPt": test_sample_count,
            "Pe_bPt": test_sample_count,
            # "Pe_nPt": test_sample_count,  # t-1p, t-2p
            "t2i": test_sample_count,
            "t3i": test_sample_count,
            "Pt_le2i": test_sample_count,
            "Pt_re2i": test_sample_count,
            "Pe_at2i": test_sample_count,
            "Pe_bt2i": test_sample_count,
            # "Pe_nt2i": test_sample_count,
            "between": test_sample_count,  # t-2i, t-3i
            "t2i_NPt": test_sample_count,
            "t2i_PtN": test_sample_count,
            "Pe_t2i_PtPe_NPt": test_sample_count,
            "t2i_N": test_sample_count,
            "t3i_N": test_sample_count,  # t-npi, t-pni, t-inp, t-2in, t-3in
            # entity
            "e2i_Pe": test_sample_count,
            "Pe_e2i": test_sample_count,  # pi, ip
            "e2u": test_sample_count,
            "Pe_e2u": test_sample_count,  # 2u, up
            # time
            "t2i_Pe": test_sample_count,
            "Pe_t2i": test_sample_count,  # t-pi, t-ip
            "t2u": test_sample_count,
            "Pe_t2u": test_sample_count,  # t-2u, t-up
        }

        def achieve_answers(train_query_structure_func, valid_query_structure_func, test_query_structure_func, for_test=False):
            answers = set()
            valid_answers = set()
            test_answers = set()
            conflict_count = -1
            placeholders = get_placeholder_list(train_query_structure_func)
            while len(answers) <= 0 or (len(answers) > 0 and (len(valid_answers) <= 0 or len(test_answers) <= 0)):
                # len(answers) > 0 and (len(valid_answers) <= 0 or len(test_answers) <= 0)
                # for queries containing negation, test may has no answers while train has lots of answers.
                # if test has no answers, we are not able to calculate metrics.
                clear_placeholder_list(placeholders)
                sampling_query_answers: FixedQuery = train_query_structure_func(*placeholders)
                if sampling_query_answers.answers is not None and len(sampling_query_answers.answers) > 0:
                    answers = sampling_query_answers.answers
                    fixed = placeholder2fixed(placeholders)
                    valid_answers = valid_query_structure_func(*fixed).answers
                    if for_test and len(valid_answers) <= len(answers) and conflict_count < 100:
                        answers = set()
                    test_answers = test_query_structure_func(*fixed).answers
                elif sampling_query_answers.timestamps is not None and len(sampling_query_answers.timestamps) > 0:
                    answers = sampling_query_answers.timestamps
                    fixed = placeholder2fixed(placeholders)
                    valid_answers = valid_query_structure_func(*fixed).timestamps
                    if for_test and len(valid_answers) <= len(answers) and conflict_count < 100:
                        answers = set()
                    test_answers = test_query_structure_func(*fixed).timestamps
                else:
                    answers = set()
                    valid_answers = set()
                    test_answers = set()
                conflict_count += 1
            # if conflict_count > 0:
            #     print("conflict_count=", conflict_count)
            queries = placeholder2sample(placeholders)
            return queries, answers, valid_answers, test_answers, conflict_count

        for query_structure_name in query_structure_name_list:
            print(query_structure_name)
            train_func = train_parser.eval(query_structure_name)
            param_name_list = get_param_name_list(train_func)
            train_queries_answers = []
            valid_queries_answers = []
            test_queries_answers = []

            fast_query_structure_name = f"fast_{query_structure_name}"
            if fast_query_structure_name in train_parser.fast_ops.keys():
                # fast sampling
                # the fast function is the proxy of the original function.
                # the fast function makes sure that len(answers)>0 with least steps (in one step if possible).
                sample_train_func = train_parser.eval(fast_query_structure_name)
            else:
                sample_train_func = train_parser.eval(query_structure_name)
            sample_valid_func = valid_parser.eval(query_structure_name)
            sample_test_func = test_parser.eval(query_structure_name)

            # 1. sampling train dataset
            if query_structure_name in train_sample_counts and query_structure_name not in self.train_queries_answers:
                sample_count = train_sample_counts[query_structure_name]
                bar = Progbar(sample_count)
                for i in range(sample_count):
                    queries, answers, valid_answers, test_answers, conflict_count = achieve_answers(
                        sample_train_func,
                        sample_valid_func,
                        sample_test_func,
                        for_test=False)
                    if None in queries:
                        raise Exception("In " + query_structure_name + ", queries contains None: " + str(queries))
                    train_queries_answers.append((queries, answers))
                    if len(valid_answers) > len(answers):
                        valid_queries_answers.append((queries, answers, valid_answers))
                    if len(test_answers) > len(answers):
                        test_queries_answers.append((queries, answers, test_answers))
                    bar.update(i + 1, {"train": len(answers), "valid": len(valid_answers), "test": len(test_answers)})
                self.train_queries_answers[query_structure_name] = {
                    "args": param_name_list,
                    "queries_answers": train_queries_answers
                }
                cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)

            # 2. sampling valid/test dataset
            if query_structure_name in test_sample_counts and query_structure_name not in self.valid_queries_answers:
                sample_count = test_sample_counts[query_structure_name]
                bar = Progbar(sample_count)
                conflict_patient = 0
                for i in range(sample_count):
                    queries, answers, valid_answers, test_answers, conflict_count = achieve_answers(
                        sample_train_func,
                        sample_valid_func,
                        sample_test_func,
                        for_test=conflict_patient <= 100)
                    if conflict_patient <= 100 and conflict_count >= 99 and i <= 1000:
                        conflict_patient += 1
                    valid_queries_answers.append((queries, answers, valid_answers))
                    test_queries_answers.append((queries, answers, test_answers))
                    bar.update(i + 1, {"train": len(answers), "valid": len(valid_answers), "test": len(test_answers)})
                    if len(valid_queries_answers) >= 10000 and len(test_queries_answers) >= 10000:
                        valid_queries_answers = valid_queries_answers[:10000]
                        test_queries_answers = test_queries_answers[:10000]
                        bar.update(sample_count, {"train": len(answers), "valid": len(valid_answers), "test": len(test_answers)})
                        break
                self.valid_queries_answers[query_structure_name] = {
                    "args": param_name_list,
                    "queries_answers": valid_queries_answers
                }
                self.test_queries_answers[query_structure_name] = {
                    "args": param_name_list,
                    "queries_answers": test_queries_answers
                }
                cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
                cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)

        # 3. calculate meta
        def avg_answers_count(qa):
            return sum([len(row[-1]) for row in qa]) / len(qa) if len(qa) > 0 else 0

        for query_name in self.test_queries_answers.keys():
            if query_name == "Pe" or query_name == "Pt":
                continue
            valid_qa = self.valid_queries_answers[query_name]["queries_answers"] if query_name in self.valid_queries_answers else []
            test_qa = self.test_queries_answers[query_name]["queries_answers"] if query_name in self.test_queries_answers else []
            self.valid_queries_answers[query_name]["queries_answers"] = valid_qa[:10000]
            self.test_queries_answers[query_name]["queries_answers"] = test_qa[:10000]
        cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
        cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)

        for query_name in self.test_queries_answers.keys():
            train_qa = self.train_queries_answers[query_name]["queries_answers"] if query_name in self.train_queries_answers else []
            valid_qa = self.valid_queries_answers[query_name]["queries_answers"] if query_name in self.valid_queries_answers else []
            test_qa = self.test_queries_answers[query_name]["queries_answers"] if query_name in self.test_queries_answers else []
            queries_answers = train_qa + valid_qa + test_qa
            self.query_meta[query_name] = {
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
            print(query_name, self.query_meta[query_name])

    def cache_all_data(self):
        TemporalKnowledgeData.cache_all_data(self)
        cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)
        cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
        cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)

    def read_meta(self, meta):
        TemporalKnowledgeData.read_meta(self, meta)
        self.query_meta = meta["query_meta"]

    def meta(self) -> Dict[str, Any]:
        m = TemporalKnowledgeData.meta(self)
        m.update({
            "query_meta": self.query_meta,
        })
        return m

    def dump(self) -> List[str]:
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = TemporalKnowledgeData.dump(self)
        for k, v in self.query_meta.items():
            dump.insert(len(dump) - 2, f"{k} : {pformat(v)}")
        return dump
