#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Set, Dict

from toolbox.data.DatasetSchema import (
    RelationalTripletDatasetSchema,
    BaseDatasetSchema,
    DBP15k,
)
from toolbox.data.functional import (
    read_cache,
    read_triple_hrt,
    read_attribute_triple_eav,
    build_map_tr_h,
    build_map_hr_t,
    read_seeds, cache_data,
)


# region 2. relational triplet data


class DatasetCachePath:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_metadata_path = self.cache_path / 'metadata.pkl'

    def is_meta_cache_exists(self):
        """ Checks if the metadata of the knowledge graph if available"""
        return self.cache_metadata_path.exists()


class RelationalTripletDatasetCachePath(DatasetCachePath):
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
        self.cache_entities_ids_path = self.cache_path / "entities_ids.pkl"
        self.cache_relations_ids_path = self.cache_path / "relations_ids.pkl"

        self.cache_idx2entity_path = self.cache_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.cache_path / 'idx2relation.pkl'
        self.cache_entity2idx_path = self.cache_path / 'entity2idx.pkl'
        self.cache_relation2idx_path = self.cache_path / 'relation2idx.pkl'

        self.cache_hr_t_path = self.cache_path / 'hr_t.pkl'
        self.cache_tr_h_path = self.cache_path / 'tr_h.pkl'
        self.cache_hr_t_train_path = self.cache_path / 'hr_t_train.pkl'
        self.cache_tr_h_train_path = self.cache_path / 'tr_h_train.pkl'
        self.cache_hr_t_valid_path = self.cache_path / 'hr_t_valid.pkl'
        self.cache_tr_h_valid_path = self.cache_path / 'tr_h_valid.pkl'
        self.cache_hr_t_test_path = self.cache_path / 'hr_t_test.pkl'
        self.cache_tr_h_test_path = self.cache_path / 'tr_h_test.pkl'

        self.cache_relation_property_path = self.cache_path / 'relation_property.pkl'


class BaseData:
    def __init__(self,
                 dataset: BaseDatasetSchema,
                 cache_path: DatasetCachePath):
        self.dataset = dataset
        self.cache_path = cache_path

    def force_prepare_data(self):
        self.read_all_origin_data()
        self.transform_all_data()
        self.cache_all_data()

    def preprocess_data_if_needed(self):
        """Function to prepare the dataset"""
        if self.is_cache_exists():
            print("data already prepared, using cache")
            return
        print("preparing data")
        self.force_prepare_data()
        print("")
        print("done!")
        print(self.print())

    def is_cache_exists(self):
        """Function to check if the dataset is cached in the memory"""
        return self.cache_path.is_meta_cache_exists()

    def read_all_origin_data(self):
        pass

    def transform_all_data(self):
        pass

    def cache_all_data(self):
        cache_data(self.meta(), self.cache_path.cache_metadata_path)

    def clear_cache(self):
        pass

    def meta(self):
        return {}

    def dump(self) -> List[str]:
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = [
            "",
            "----------Metadata Info for Dataset:%s----------------" % self.dataset.name,
            "---------------------------------------------",
            "",
        ]
        return dump

    def print(self, log=print):
        for i in self.dump():
            log(i)


class RelationalTripletData(BaseData):
    """ The class is the main module that handles the knowledge graph.

        KnowledgeGraph is responsible for downloading, parsing, processing and preparing
        the training, testing and validation dataset.

        Args:
            dataset (RelationalTripletDatasetSchema): custom dataset.
            cache_path (RelationalTripletDatasetCachePath): cache path.

        Attributes:
            dataset (object): The dataset object isntance.
            all_relations (list):list of all the relations.
            all_entities (list): List of all the entities.
            entity2idx (dict): Dictionary for mapping string name of entities to unique numerical id.
            idx2entity (dict): Dictionary for mapping the id to string.
            relation2idx (dict): Dictionary for mapping the id to string.
            idx2relation (dict): Dictionary for mapping the id to string.
            hr_t (dict):  Dictionary with set as a default key and list as values.
            tr_h (dict):  Dictionary with set as a default key and list as values.
            hr_t_train (dict):  Dictionary with set as a default key and list as values.
            tr_h_train (dict):  Dictionary with set as a default key and list as values.
            relation_property (list): list storing the entities tied to a specific relation.

        Examples:
            >>> from toolbox.data.DataSchema import RelationalTripletData, RelationalTripletDatasetCachePath
            >>> from toolbox.data.DatasetSchema import FreebaseFB15k_237
            >>> dataset = FreebaseFB15k_237()
            >>> cache = RelationalTripletDatasetCachePath(dataset.cache_path)
            >>> data = RelationalTripletData(dataset=dataset, cache_path=cache)
            >>> data.preprocess_data_if_needed()

    """

    def __init__(self,
                 dataset: RelationalTripletDatasetSchema,
                 cache_path: RelationalTripletDatasetCachePath):
        BaseData.__init__(self, dataset, cache_path)
        self.dataset = dataset
        self.cache_path = cache_path

        # KG data structure stored in triplet format
        self.all_triples: List[Tuple[str, str, str]] = []
        self.train_triples: List[Tuple[str, str, str]] = []
        self.test_triples: List[Tuple[str, str, str]] = []
        self.valid_triples: List[Tuple[str, str, str]] = []

        self.all_triples_ids: List[Tuple[int, int, int]] = []
        self.train_triples_ids: List[Tuple[int, int, int]] = []
        self.test_triples_ids: List[Tuple[int, int, int]] = []
        self.valid_triples_ids: List[Tuple[int, int, int]] = []

        self.all_relations: List[str] = []
        self.all_entities: List[str] = []
        self.entities_ids: List[int] = []
        self.relations_ids: List[int] = []

        self.entity2idx: Dict[str, int] = {}
        self.idx2entity: Dict[int, str] = {}
        self.relation2idx: Dict[str, int] = {}
        self.idx2relation: Dict[int, str] = {}

        self.hr_t: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.tr_h: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        self.hr_t_train: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.tr_h_train: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        self.hr_t_valid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.tr_h_valid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        self.hr_t_test: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.tr_h_test: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        self.relation_property: Dict[int, float] = {}

        # meta
        self.entity_count = 0
        self.relation_count = 0
        self.valid_triples_count = 0
        self.test_triples_count = 0
        self.train_triples_count = 0
        self.triple_count = 0

    def read_all_origin_data(self):
        self.read_all_triplets()

    def read_all_triplets(self):
        self.train_triples = read_triple_hrt(self.dataset.data_paths['train'])
        self.valid_triples = read_triple_hrt(self.dataset.data_paths['valid'])
        self.test_triples = read_triple_hrt(self.dataset.data_paths['test'])
        self.all_triples = self.train_triples + self.valid_triples + self.test_triples

        self.valid_triples_count = len(self.valid_triples)
        self.test_triples_count = len(self.test_triples)
        self.train_triples_count = len(self.train_triples)
        self.triple_count = self.valid_triples_count + self.test_triples_count + self.train_triples_count

    def transform_all_data(self):
        self.transform_entities_relations()
        self.transform_mappings()
        self.transform_all_triplets_ids()

        self.transform_entity_ids()
        self.transform_relation_ids()

        self.transform_hr_t()
        self.transform_tr_h()
        self.transform_hr_t_train()
        self.transform_tr_h_train()
        self.transform_hr_t_valid()
        self.transform_tr_h_valid()
        self.transform_hr_t_test()
        self.transform_tr_h_test()

        self.transform_relation_property()

    def transform_entities_relations(self):
        """ Function to read the entities. """
        entities: Set[str] = set()
        relations: Set[str] = set()
        # print("entities_relations")
        # bar = Progbar(len(self.all_triples))
        # i = 0
        for h, r, t in self.all_triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
            # i += 1
            # bar.update(i, [("h", h.split("/")[-1]), ("r", r.split("/")[-1]), ("t", t.split("/")[-1])])

        self.all_entities = sorted(list(entities))
        self.all_relations = sorted(list(relations))

        self.entity_count = len(self.all_entities)
        self.relation_count = len(self.all_relations)

    def transform_mappings(self):
        """ Function to generate the mapping from string name to integer ids. """
        self.entity2idx = {v: k for k, v in enumerate(self.all_entities)}
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        self.relation2idx = {v: k for k, v in enumerate(self.all_relations)}
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}

    def transform_all_triplets_ids(self):
        entity2idx = self.entity2idx
        relation2idx = self.relation2idx
        self.train_triples_ids = [(entity2idx[h], relation2idx[r], entity2idx[t]) for h, r, t in self.train_triples]
        self.test_triples_ids = [(entity2idx[h], relation2idx[r], entity2idx[t]) for h, r, t in self.test_triples]
        self.valid_triples_ids = [(entity2idx[h], relation2idx[r], entity2idx[t]) for h, r, t in self.valid_triples]
        self.all_triples_ids = self.train_triples_ids + self.valid_triples_ids + self.test_triples_ids

    def transform_entity_ids(self):

        entity2idx = self.entity2idx

        print("entities_ids")
        # bar = Progbar(len(self.all_entities))
        # i = 0
        for e in self.all_entities:
            self.entities_ids.append(entity2idx[e])
            # i += 1
            # bar.update(i, [("entity", e.split("/")[-1])])

    def transform_relation_ids(self):

        relation2idx = self.relation2idx

        print("relations_ids")
        # bar = Progbar(len(self.all_relations))
        # i = 0
        for r in self.all_relations:
            self.relations_ids.append(relation2idx[r])
            # i += 1
            # bar.update(i, [("relation", r.split("/")[-1])])

    def transform_hr_t(self):
        """ Function to read the list of tails for the given head and relation pair. """
        self.hr_t = build_map_hr_t(self.all_triples_ids)

    def transform_tr_h(self):
        """ Function to read the list of heads for the given tail and relation pair. """
        self.tr_h = build_map_tr_h(self.all_triples_ids)

    def transform_hr_t_train(self):
        """ Function to read the list of tails for the given head and relation pair for the training set. """
        self.hr_t_train = build_map_hr_t(self.train_triples_ids)

    def transform_tr_h_train(self):
        """ Function to read the list of heads for the given tail and relation pair for the training set. """
        self.tr_h_train = build_map_tr_h(self.train_triples_ids)

    def transform_hr_t_valid(self):
        """ Function to read the list of tails for the given head and relation pair for the valid set. """
        self.hr_t_valid = build_map_hr_t(self.valid_triples_ids)

    def transform_tr_h_valid(self):
        """ Function to read the list of heads for the given tail and relation pair for the valid set. """
        self.tr_h_valid = build_map_tr_h(self.valid_triples_ids)

    def transform_hr_t_test(self):
        """ Function to read the list of tails for the given head and relation pair for the valid set. """
        self.hr_t_test = build_map_hr_t(self.test_triples_ids)

    def transform_tr_h_test(self):
        """ Function to read the list of heads for the given tail and relation pair for the valid set. """
        self.tr_h_test = build_map_tr_h(self.test_triples_ids)

    def transform_relation_property(self):
        """ Function to read the relation property.

         Returns:
             list: Returns the list of relation property.
         """
        relation_property_head = {x: [] for x in range(len(self.all_relations))}
        relation_property_tail = {x: [] for x in range(len(self.all_relations))}

        # print("relation_property")
        # bar = Progbar(len(self.train_triples_ids))
        # i = 0
        for h, r, t in self.train_triples_ids:
            relation_property_head[r].append(h)
            relation_property_tail[r].append(t)
            # i += 1
            # bar.update(i, [])

        self.relation_property = {}
        for x in relation_property_head.keys():
            value_up = len(set(relation_property_tail[x]))

            value_bot = len(set(relation_property_head[x])) + len(set(relation_property_tail[x]))

            if value_bot == 0:
                value = 0
            else:
                value = value_up / value_bot

            self.relation_property[x] = value

        return self.relation_property

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
        cache_data(self.entities_ids, self.cache_path.cache_entities_ids_path)
        cache_data(self.relations_ids, self.cache_path.cache_relations_ids_path)

        cache_data(self.idx2entity, self.cache_path.cache_idx2entity_path)
        cache_data(self.idx2relation, self.cache_path.cache_idx2relation_path)
        cache_data(self.relation2idx, self.cache_path.cache_relation2idx_path)
        cache_data(self.entity2idx, self.cache_path.cache_entity2idx_path)

        cache_data(self.hr_t, self.cache_path.cache_hr_t_path)
        cache_data(self.tr_h, self.cache_path.cache_tr_h_path)
        cache_data(self.hr_t_train, self.cache_path.cache_hr_t_train_path)
        cache_data(self.tr_h_train, self.cache_path.cache_tr_h_train_path)
        cache_data(self.hr_t_valid, self.cache_path.cache_hr_t_valid_path)
        cache_data(self.tr_h_valid, self.cache_path.cache_tr_h_valid_path)
        cache_data(self.hr_t_test, self.cache_path.cache_hr_t_test_path)
        cache_data(self.tr_h_test, self.cache_path.cache_tr_h_test_path)

        cache_data(self.relation_property, self.cache_path.cache_relation_property_path)

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
        self.relation_count = meta["relation_count"]
        self.entity_count = meta["entity_count"]
        self.valid_triples_count = meta["valid_triples_count"]
        self.test_triples_count = meta["test_triples_count"]
        self.train_triples_count = meta["train_triples_count"]
        self.triple_count = meta["triple_count"]

    def meta(self):
        return {
            "relation_count": self.relation_count,
            "entity_count": self.entity_count,
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
            "----------Metadata Info for Dataset:%s----------------" % self.dataset.name,
            "Total Training Triples   :%s" % self.train_triples_count,
            "Total Testing Triples    :%s" % self.test_triples_count,
            "Total validation Triples :%s" % self.valid_triples_count,
            "Total Entities           :%s" % self.entity_count,
            "Total Relations          :%s" % self.relation_count,
            "---------------------------------------------",
            "",
        ]
        return dump


# endregion


class DBP15kCachePath(RelationalTripletDatasetCachePath):
    def __init__(self, cache_path: Path):
        RelationalTripletDatasetCachePath.__init__(self, cache_path)
        self.cache_entity_align_file = self.cache_path / "ref_ent_ids.txt"
        self.cache_all_entity_file = self.cache_path / "ent_ids_all.txt"
        self.cache_all_attr_file = self.cache_path / "att2id_all.txt"
        self.cache_all_value_file = self.cache_path / "att_value2id_all.txt"
        self.cache_all_triple_file = self.cache_path / "triples_struct_all.txt"
        self.cache_all_triple_file_enhance = self.cache_path / "triples_struct_all_enhance.txt"
        self.cache_kg1_entity_file = self.cache_path / "ent_ids_1.txt"
        self.cache_kg2_entity_file = self.cache_path / "ent_ids_2.txt"

        self.cache_seeds_path = self.cache_path / "seeds.pkl"
        self.cache_train_seeds_path = self.cache_path / "train_seeds.pkl"
        self.cache_test_seeds_path = self.cache_path / "test_seeds.pkl"
        self.cache_seeds_ids_path = self.cache_path / "seeds_ids.pkl"
        self.cache_train_seeds_ids_path = self.cache_path / "train_seeds_ids.pkl"
        self.cache_test_seeds_ids_path = self.cache_path / "test_seeds_ids.pkl"
        self.cache_left_ids_path = self.cache_path / "left_ids.pkl"
        self.cache_right_ids_path = self.cache_path / "right_ids.pkl"

        self.cache_kg1_triples_path = self.cache_path / "kg1_triples.pkl"
        self.cache_kg2_triples_path = self.cache_path / "kg2_triples.pkl"
        self.cache_kg1_triples_ids_path = self.cache_path / "kg1_triples_ids.pkl"
        self.cache_kg2_triples_ids_path = self.cache_path / "kg2_triples_ids.pkl"
        self.cache_all_attribute_triples_path = self.cache_path / "all_attribute_triples.pkl"
        self.cache_kg1_attribute_triples_path = self.cache_path / "kg1_attribute_triples.pkl"
        self.cache_kg2_attribute_triples_path = self.cache_path / "kg2_attribute_triples.pkl"
        self.cache_all_attribute_triples_ids_path = self.cache_path / "all_attribute_triples_ids.pkl"
        self.cache_kg1_attribute_triples_ids_path = self.cache_path / "kg1_attribute_triples_ids.pkl"
        self.cache_kg2_attribute_triples_ids_path = self.cache_path / "kg2_attribute_triples_ids.pkl"
        self.cache_all_attribute_names_path = self.cache_path / "all_attribute_names.pkl"
        self.cache_all_attribute_values_path = self.cache_path / "all_attribute_values.pkl"

        self.cache_kg1_entities_path = self.cache_path / "kg1_entities.pkl"
        self.cache_kg2_entities_path = self.cache_path / "kg2_entities.pkl"
        self.cache_kg1_entities_ids_path = self.cache_path / "kg1_entities_ids.pkl"
        self.cache_kg2_entities_ids_path = self.cache_path / "kg2_entities_ids.pkl"
        self.cache_kg1_relations_path = self.cache_path / "kg1_relations.pkl"
        self.cache_kg2_relations_path = self.cache_path / "kg2_relations.pkl"
        self.cache_kg1_relations_ids_path = self.cache_path / "kg1_relations_ids.pkl"
        self.cache_kg2_relations_ids_path = self.cache_path / "kg2_relations_ids.pkl"
        self.cache_kg1_attribute_names_path = self.cache_path / "kg1_attribute_names.pkl"
        self.cache_kg2_attribute_names_path = self.cache_path / "kg2_attribute_names.pkl"
        self.cache_kg1_attribute_names_ids_path = self.cache_path / "kg1_attribute_names_ids.pkl"
        self.cache_kg2_attribute_names_ids_path = self.cache_path / "kg2_attribute_names_ids.pkl"
        self.cache_attribute_names_ids_path = self.cache_path / "attribute_names_ids.pkl"
        self.cache_kg1_attribute_values_path = self.cache_path / "kg1_attribute_values.pkl"
        self.cache_kg2_attribute_values_path = self.cache_path / "kg2_attribute_values.pkl"
        self.cache_kg1_attribute_values_ids_path = self.cache_path / "kg1_attribute_values_ids.pkl"
        self.cache_kg2_attribute_values_ids_path = self.cache_path / "kg2_attribute_values_ids.pkl"
        self.cache_attribute_values_ids_path = self.cache_path / "attribute_values_ids.pkl"

        self.cache_attribute_name2idx_path = self.cache_path / "attribute_name2idx.pkl"
        self.cache_idx2attribute_name_path = self.cache_path / "idx2attribute_name.pkl"
        self.cache_attribute_value2idx_path = self.cache_path / "attribute_value2idx.pkl"
        self.cache_idx2attribute_value_path = self.cache_path / "idx2attribute_value.pkl"


class DBP15kData(RelationalTripletData):
    def __init__(self,
                 dataset: DBP15k,
                 cache_path: DBP15kCachePath,
                 train_seeds_percent=0.3):
        RelationalTripletData.__init__(self, dataset, cache_path)
        self.cache_path = cache_path
        self.train_seeds_percent = train_seeds_percent

        self.kg1_triples: List[Tuple[str, str, str]] = []
        self.kg2_triples: List[Tuple[str, str, str]] = []

        self.kg1_triples_ids: List[Tuple[int, int, int]] = []
        self.kg2_triples_ids: List[Tuple[int, int, int]] = []

        self.all_attribute_triples: List[Tuple[str, str, str]] = []
        self.kg1_attribute_triples: List[Tuple[str, str, str]] = []
        self.kg2_attribute_triples: List[Tuple[str, str, str]] = []

        self.all_attribute_triples_ids: List[Tuple[int, int, int]] = []
        self.kg1_attribute_triples_ids: List[Tuple[int, int, int]] = []
        self.kg2_attribute_triples_ids: List[Tuple[int, int, int]] = []

        self.all_attribute_names: List[str] = []
        self.all_attribute_values: List[str] = []
        self.all_attribute_names_ids: List[int] = []
        self.all_attribute_values_ids: List[int] = []

        self.kg1_entities: List[str] = []
        self.kg2_entities: List[str] = []
        self.kg1_entities_ids: List[int] = []
        self.kg2_entities_ids: List[int] = []
        self.entities_ids: List[int] = []

        self.kg1_relations: List[str] = []
        self.kg2_relations: List[str] = []
        self.kg1_relations_ids: List[int] = []
        self.kg2_relations_ids: List[int] = []
        self.relations_ids: List[int] = []

        self.kg1_attribute_names: List[str] = []
        self.kg2_attribute_names: List[str] = []
        self.kg1_attribute_names_ids: List[int] = []
        self.kg2_attribute_names_ids: List[int] = []
        self.attribute_names_ids: List[int] = []

        self.kg1_attribute_values: List[str] = []
        self.kg2_attribute_values: List[str] = []
        self.kg1_attribute_values_ids: List[int] = []
        self.kg2_attribute_values_ids: List[int] = []
        self.attribute_values_ids: List[int] = []

        self.attribute_name2idx: Dict[str, int] = {}
        self.idx2attribute_name: Dict[int, str] = {}
        self.attribute_value2idx: Dict[str, int] = {}
        self.idx2attribute_value: Dict[int, str] = {}

        self.seeds: List[Tuple[str, str]] = []  # (m, 2) m个对齐的实体对(a,b)称a为左实体，b为右实体
        self.train_seeds: List[Tuple[str, str]] = []  # (0.3m, 2)
        self.test_seeds: List[Tuple[str, str]] = []  # (0.7m, 2)
        self.seeds_ids: List[Tuple[int, int]] = []  # (m, 2) m个对齐的实体对(a,b)称a为左实体，b为右实体
        self.train_seeds_ids: List[Tuple[int, int]] = []  # (0.3m, 2)
        self.test_seeds_ids: List[Tuple[int, int]] = []  # (0.7m, 2)
        self.left_ids: List[int] = []  # test_seeds 中对齐实体的左实体id
        self.right_ids: List[int] = []  # test_seeds 中对齐实体的右实体id

        self.kg1_triples_count = 0
        self.kg2_triples_count = 0

        self.all_attribute_triples_count = 0
        self.kg1_attribute_triples_count = 0
        self.kg2_attribute_triples_count = 0

        self.alignment_seeds_count = 0
        self.valid_alignment_seeds_count = 0
        self.test_alignment_seeds_count = 0
        self.train_alignment_seeds_count = 0

        self.kg1_entities_count = 0
        self.kg2_entities_count = 0

        self.kg1_relations_count = 0
        self.kg2_relations_count = 0

        self.all_attribute_names_count = 0
        self.kg1_attribute_names_count = 0
        self.kg2_attribute_names_count = 0

        self.all_attribute_values_count = 0
        self.kg1_attribute_values_count = 0
        self.kg2_attribute_values_count = 0

    def read_all_origin_data(self):
        self.read_all_triplets()
        self.read_attribute_triplets()
        self.read_entity_align_list()

    def read_all_triplets(self):
        self.kg1_triples = read_triple_hrt(self.dataset.data_paths['kg1_relational_triples'])
        self.kg2_triples = read_triple_hrt(self.dataset.data_paths['kg2_relational_triples'])
        self.all_triples = self.kg1_triples + self.kg2_triples
        self.train_triples = self.all_triples
        self.test_triples = []
        self.valid_triples = []

        self.kg1_triples_count = len(self.kg1_triples)
        self.kg2_triples_count = len(self.kg2_triples)
        self.triple_count = len(self.all_attribute_triples)
        self.train_triples_count = len(self.train_triples)
        self.test_triples_count = len(self.test_triples)
        self.valid_triples_count = len(self.valid_triples)

    def read_attribute_triplets(self):
        self.kg1_attribute_triples = read_attribute_triple_eav(self.dataset.data_paths['kg1_attribute_triples'])
        self.kg2_attribute_triples = read_attribute_triple_eav(self.dataset.data_paths['kg2_attribute_triples'])
        self.all_attribute_triples = self.kg1_attribute_triples + self.kg2_attribute_triples

        self.kg1_attribute_triples_count = len(self.kg1_attribute_triples)
        self.kg2_attribute_triples_count = len(self.kg2_attribute_triples)
        self.all_attribute_triples_count = len(self.all_attribute_triples)

    def read_entity_align_list(self):
        self.seeds = read_seeds(self.dataset.data_paths['seeds'])
        train_max_idx = int(self.train_seeds_percent * len(self.seeds))

        self.train_seeds = self.seeds[:train_max_idx]
        self.test_seeds = self.seeds[train_max_idx:]

        self.alignment_seeds_count = len(self.seeds)
        self.valid_alignment_seeds_count = 0
        self.test_alignment_seeds_count = len(self.test_seeds)
        self.train_alignment_seeds_count = len(self.train_seeds)

    def transform_all_data(self):
        RelationalTripletData.transform_all_data(self)
        self.transform_attribute_names_values()
        self.transform_attribute_mappings()
        self.transform_all_attribute_triplets_ids()

        self.transform_attribute_name_ids()
        self.transform_attribute_value_ids()
        self.transform_entity_align_ids()

    def transform_entities_relations(self):
        entities: Set[str] = set()
        kg1_entities: Set[str] = set()
        kg2_entities: Set[str] = set()
        relations: Set[str] = set()
        kg1_relations: Set[str] = set()
        kg2_relations: Set[str] = set()

        print("kg1_entities_relations")
        # bar = Progbar(len(self.kg1_triples))
        # i = 0
        for h, r, t in self.kg1_triples:
            kg1_entities.add(h)
            kg1_entities.add(t)
            kg1_relations.add(r)
            # i += 1
            # bar.update(i, [("h", h.split("/")[-1]), ("r", r.split("/")[-1]), ("t", t.split("/")[-1])])

        print("kg2_entities_relations")
        # bar = Progbar(len(self.kg2_triples))
        # i = 0
        for h, r, t in self.kg2_triples:
            kg2_entities.add(h)
            kg2_entities.add(t)
            kg2_relations.add(r)
            # i += 1
            # bar.update(i, [("h", h.split("/")[-1]), ("r", r.split("/")[-1]), ("t", t.split("/")[-1])])
        entities = kg2_entities.union(kg1_entities)
        relations = kg2_relations.union(kg1_relations)

        self.all_entities = sorted(list(entities))
        self.kg1_entities = sorted(list(kg1_entities))
        self.kg2_entities = sorted(list(kg2_entities))

        self.all_relations = sorted(list(relations))
        self.kg1_relations = sorted(list(kg1_relations))
        self.kg2_relations = sorted(list(kg2_relations))

        self.entity_count = len(self.all_entities)
        self.kg1_entities_count = len(self.kg1_entities)
        self.kg2_entities_count = len(self.kg2_entities)

        self.relation_count = len(self.all_relations)
        self.kg1_relations_count = len(self.kg1_relations)
        self.kg2_relations_count = len(self.kg2_relations)

    def transform_attribute_names_values(self):
        attribute_names: Set[str] = set()
        kg1_attribute_names: Set[str] = set()
        kg2_attribute_names: Set[str] = set()
        attribute_values: Set[str] = set()
        kg1_attribute_values: Set[str] = set()
        kg2_attribute_values: Set[str] = set()

        print("kg1_attribute_names_values")
        # bar = Progbar(len(self.kg1_attribute_triples))
        # i = 0
        for e, a, v in self.kg1_attribute_triples:
            kg1_attribute_names.add(a)
            kg1_attribute_values.add(v)
            # i += 1
            # bar.update(i, [("name", a.split("/")[-1]), ("value", v)])

        print("kg2_attribute_names_values")
        # bar = Progbar(len(self.kg2_attribute_triples))
        # i = 0
        for e, a, v in self.kg2_attribute_triples:
            kg2_attribute_names.add(a)
            kg2_attribute_values.add(v)
            # i += 1
            # bar.update(i, [("name", a.split("/")[-1]), ("value", v)])
        attribute_names = kg1_attribute_names.union(kg2_attribute_names)
        attribute_values = kg1_attribute_values.union(kg2_attribute_values)
        self.all_attribute_names = sorted(list(attribute_names))
        self.kg1_attribute_names = sorted(list(kg1_attribute_names))
        self.kg2_attribute_names = sorted(list(kg2_attribute_names))

        self.all_attribute_names_count = len(self.all_attribute_names)
        self.kg1_attribute_names_count = len(self.kg1_attribute_names)
        self.kg2_attribute_names_count = len(self.kg2_attribute_names)

        self.all_attribute_values = sorted(list(attribute_values))
        self.kg1_attribute_values = sorted(list(kg1_attribute_values))
        self.kg2_attribute_values = sorted(list(kg2_attribute_values))

        self.all_attribute_values_count = len(self.all_attribute_values)
        self.kg1_attribute_values_count = len(self.kg1_attribute_values)
        self.kg2_attribute_values_count = len(self.kg2_attribute_values)

    def transform_attribute_mappings(self):
        """ Function to generate the mapping from string name to integer ids. """
        self.idx2attribute_name = {k: v for k, v in enumerate(self.all_attribute_names)}
        self.attribute_name2idx = {v: k for k, v in self.idx2attribute_name.items()}
        self.idx2attribute_value = {k: v for k, v in enumerate(self.all_attribute_values)}
        self.attribute_value2idx = {v: k for k, v in self.idx2attribute_value.items()}

    def transform_all_triplets_ids(self):
        entity2idx = self.entity2idx
        relation2idx = self.relation2idx

        print("kg1_triples_ids")
        # bar = Progbar(len(self.kg1_triples))
        # i = 0
        for h, r, t in self.kg1_triples:
            self.kg1_triples_ids.append((entity2idx[h], relation2idx[r], entity2idx[t]))
            # i += 1
            # bar.update(i, [("h", h.split("/")[-1]), ("r", r.split("/")[-1]), ("t", t.split("/")[-1])])

        print("kg2_triples_ids")
        # bar = Progbar(len(self.kg2_triples))
        # i = 0
        for h, r, t in self.kg2_triples:
            self.kg2_triples_ids.append((entity2idx[h], relation2idx[r], entity2idx[t]))
            # i += 1
            # bar.update(i, [("h", h.split("/")[-1]), ("r", r.split("/")[-1]), ("t", t.split("/")[-1])])

        self.all_triples_ids = self.kg1_triples_ids + self.kg2_triples_ids
        self.train_triples_ids = self.all_triples_ids
        self.test_triples_ids = []
        self.valid_triples_ids = []

    def transform_entity_align_ids(self):
        entity2idx = self.entity2idx
        for left_entity, right_entity in self.seeds:
            self.seeds_ids.append((entity2idx[left_entity], entity2idx[right_entity]))
        train_max_idx = int(self.train_seeds_percent * len(self.seeds))
        self.train_seeds_ids = self.seeds_ids[:train_max_idx]
        self.test_seeds_ids = self.seeds_ids[train_max_idx:]

        self.left_ids = []
        self.right_ids = []
        for left_entity, right_entity in self.test_seeds_ids:
            self.left_ids.append(left_entity)  # 对齐的左边的实体
            self.right_ids.append(right_entity)  # 对齐的右边的实体

    def transform_all_attribute_triplets_ids(self):
        entity2idx = self.entity2idx
        attribute_name2idx = self.attribute_name2idx
        attribute_value2idx = self.attribute_value2idx

        print("kg1_attribute_triples_ids")
        # bar = Progbar(len(self.kg1_attribute_triples))
        # i = 0
        for e, a, v in self.kg1_attribute_triples:
            self.kg1_attribute_triples_ids.append((entity2idx[e], attribute_name2idx[a], attribute_value2idx[v]))
            # i += 1
            # bar.update(i, [("e", e.split("/")[-1]), ("a", a.split("/")[-1]), ("v", v)])

        print("kg2_attribute_triples_ids")
        # bar = Progbar(len(self.kg2_attribute_triples))
        # i = 0
        for e, a, v in self.kg2_attribute_triples:
            self.kg2_attribute_triples_ids.append((entity2idx[e], attribute_name2idx[a], attribute_value2idx[v]))
            # i += 1
            # bar.update(i, [("e", e.split("/")[-1]), ("a", a.split("/")[-1]), ("v", v)])

        self.all_attribute_triples_ids = self.kg1_attribute_triples_ids + self.kg2_attribute_triples_ids

    def transform_entity_ids(self):

        entity2idx = self.entity2idx

        print("kg1_entities_ids")
        # bar = Progbar(len(self.kg1_entities))
        # i = 0
        for e in self.kg1_entities:
            self.kg1_entities_ids.append(entity2idx[e])
            # i += 1
            # bar.update(i, [("entity", e.split("/")[-1])])

        print("kg2_entities_ids")
        # bar = Progbar(len(self.kg2_entities))
        # i = 0
        for e in self.kg2_entities:
            self.kg2_entities_ids.append(entity2idx[e])
            # i += 1
            # bar.update(i, [("entity", e.split("/")[-1])])

        self.entities_ids = self.kg1_entities_ids + self.kg2_entities_ids

    def transform_relation_ids(self):

        relation2idx = self.relation2idx

        print("kg1_relations_ids")
        # bar = Progbar(len(self.kg1_relations))
        # i = 0
        for r in self.kg1_relations:
            self.kg1_relations_ids.append(relation2idx[r])
            # i += 1
            # bar.update(i, [("relation", r.split("/")[-1])])

        print("kg2_relations_ids")
        # bar = Progbar(len(self.kg2_relations))
        # i = 0
        for r in self.kg2_relations:
            self.kg2_relations_ids.append(relation2idx[r])
            # i += 1
            # bar.update(i, [("relation", r.split("/")[-1])])

        self.relations_ids = self.kg1_relations_ids + self.kg2_relations_ids

    def transform_attribute_name_ids(self):

        attribute_name2idx = self.attribute_name2idx

        print("kg1_attribute_names_ids")
        # bar = Progbar(len(self.kg1_attribute_names))
        # i = 0
        for r in self.kg1_attribute_names:
            self.kg1_attribute_names_ids.append(attribute_name2idx[r])
            # i += 1
            # bar.update(i, [("attribute_names", r.split("/")[-1])])

        print("kg2_attribute_names_ids")
        # bar = Progbar(len(self.kg2_attribute_names))
        # i = 0
        for r in self.kg2_attribute_names:
            self.kg2_attribute_names_ids.append(attribute_name2idx[r])
            # i += 1
            # bar.update(i, [("attribute_names", r.split("/")[-1])])

        self.attribute_names_ids = self.kg1_attribute_names_ids + self.kg2_attribute_names_ids

    def transform_attribute_value_ids(self):

        attribute_value2idx = self.attribute_value2idx

        print("kg1_attribute_values_ids")
        # bar = Progbar(len(self.kg1_attribute_values))
        # i = 0
        for r in self.kg1_attribute_values:
            self.kg1_attribute_values_ids.append(attribute_value2idx[r])
            # i += 1
            # bar.update(i, [("attribute_value", r)])

        print("kg2_attribute_values_ids")
        # bar = Progbar(len(self.kg2_attribute_values))
        # i = 0
        for r in self.kg2_attribute_values:
            self.kg2_attribute_values_ids.append(attribute_value2idx[r])
            # i += 1
            # bar.update(i, [("attribute_value", r)])

        self.attribute_values_ids = self.kg1_attribute_values_ids + self.kg2_attribute_values_ids

    def cache_all_data(self):
        cache_data(self.kg1_triples, self.cache_path.cache_kg1_triples_path)
        cache_data(self.kg2_triples, self.cache_path.cache_kg2_triples_path)
        cache_data(self.kg1_triples_ids, self.cache_path.cache_kg1_triples_ids_path)
        cache_data(self.kg2_triples_ids, self.cache_path.cache_kg2_triples_ids_path)
        cache_data(self.all_attribute_triples, self.cache_path.cache_all_attribute_triples_path)
        cache_data(self.kg1_attribute_triples, self.cache_path.cache_kg1_attribute_triples_path)
        cache_data(self.kg2_attribute_triples, self.cache_path.cache_kg2_attribute_triples_path)
        cache_data(self.all_attribute_triples_ids, self.cache_path.cache_all_attribute_triples_ids_path)
        cache_data(self.kg1_attribute_triples_ids, self.cache_path.cache_kg1_attribute_triples_ids_path)
        cache_data(self.kg2_attribute_triples_ids, self.cache_path.cache_kg2_attribute_triples_ids_path)
        cache_data(self.all_attribute_names, self.cache_path.cache_all_attribute_names_path)
        cache_data(self.all_attribute_values, self.cache_path.cache_all_attribute_values_path)

        cache_data(self.kg1_entities, self.cache_path.cache_kg1_entities_path)
        cache_data(self.kg2_entities, self.cache_path.cache_kg2_entities_path)
        cache_data(self.kg1_entities_ids, self.cache_path.cache_kg1_entities_ids_path)
        cache_data(self.kg2_entities_ids, self.cache_path.cache_kg2_entities_ids_path)
        cache_data(self.entities_ids, self.cache_path.cache_entities_ids_path)
        cache_data(self.kg1_relations, self.cache_path.cache_kg1_relations_path)
        cache_data(self.kg2_relations, self.cache_path.cache_kg2_relations_path)
        cache_data(self.kg1_relations_ids, self.cache_path.cache_kg1_relations_ids_path)
        cache_data(self.kg2_relations_ids, self.cache_path.cache_kg2_relations_ids_path)
        cache_data(self.relations_ids, self.cache_path.cache_relations_ids_path)
        cache_data(self.kg1_attribute_names, self.cache_path.cache_kg1_attribute_names_path)
        cache_data(self.kg2_attribute_names, self.cache_path.cache_kg2_attribute_names_path)
        cache_data(self.kg1_attribute_names_ids, self.cache_path.cache_kg1_attribute_names_ids_path)
        cache_data(self.kg2_attribute_names_ids, self.cache_path.cache_kg2_attribute_names_ids_path)
        cache_data(self.attribute_names_ids, self.cache_path.cache_attribute_names_ids_path)
        cache_data(self.kg1_attribute_values, self.cache_path.cache_kg1_attribute_values_path)
        cache_data(self.kg2_attribute_values, self.cache_path.cache_kg2_attribute_values_path)
        cache_data(self.kg1_attribute_values_ids, self.cache_path.cache_kg1_attribute_values_ids_path)
        cache_data(self.kg2_attribute_values_ids, self.cache_path.cache_kg2_attribute_values_ids_path)
        cache_data(self.attribute_values_ids, self.cache_path.cache_attribute_values_ids_path)

        cache_data(self.attribute_name2idx, self.cache_path.cache_attribute_name2idx_path)
        cache_data(self.idx2attribute_name, self.cache_path.cache_idx2attribute_name_path)
        cache_data(self.attribute_value2idx, self.cache_path.cache_attribute_value2idx_path)
        cache_data(self.idx2attribute_value, self.cache_path.cache_idx2attribute_value_path)

        cache_data(self.seeds, self.cache_path.cache_seeds_path)
        cache_data(self.train_seeds, self.cache_path.cache_train_seeds_path)
        cache_data(self.test_seeds, self.cache_path.cache_test_seeds_path)
        cache_data(self.seeds_ids, self.cache_path.cache_seeds_ids_path)
        cache_data(self.train_seeds_ids, self.cache_path.cache_train_seeds_ids_path)
        cache_data(self.test_seeds_ids, self.cache_path.cache_test_seeds_ids_path)
        cache_data(self.left_ids, self.cache_path.cache_left_ids_path)
        cache_data(self.right_ids, self.cache_path.cache_right_ids_path)
        RelationalTripletData.cache_all_data(self)

    def read_meta(self, meta):
        self.entity_count = meta["entity_count"]
        self.relation_count = meta["relation_count"]
        self.triple_count = meta["triple_count"]
        self.train_triples_count = meta["train_triples_count"]
        self.test_triples_count = meta["test_triples_count"]
        self.valid_triples_count = meta["valid_triples_count"]

        self.kg1_triples_count = meta["kg1_triples_count"]
        self.kg2_triples_count = meta["kg2_triples_count"]
        self.all_attribute_triples_count = meta["all_attribute_triples_count"]
        self.kg1_attribute_triples_count = meta["kg1_attribute_triples_count"]
        self.kg2_attribute_triples_count = meta["kg2_attribute_triples_count"]
        self.alignment_seeds_count = meta["alignment_seeds_count"]
        self.valid_alignment_seeds_count = meta["valid_alignment_seeds_count"]
        self.test_alignment_seeds_count = meta["test_alignment_seeds_count"]
        self.train_alignment_seeds_count = meta["train_alignment_seeds_count"]
        self.all_attribute_names_count = meta["all_attribute_names_count"]
        self.all_attribute_values_count = meta["all_attribute_values_count"]
        self.kg1_entities_count = meta["kg1_entities_count"]
        self.kg2_entities_count = meta["kg2_entities_count"]
        self.kg1_relations_count = meta["kg1_relations_count"]
        self.kg2_relations_count = meta["kg2_relations_count"]
        self.kg1_attribute_names_count = meta["kg1_attribute_names_count"]
        self.kg2_attribute_names_count = meta["kg2_attribute_names_count"]
        self.kg1_attribute_values_count = meta["kg1_attribute_values_count"]
        self.kg2_attribute_values_count = meta["kg2_attribute_values_count"]

    def meta(self):
        return {
            "dataset": self.dataset.name,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "triple_count": self.triple_count,
            "train_triples_count": self.train_triples_count,
            "test_triples_count": self.test_triples_count,
            "valid_triples_count": self.valid_triples_count,

            "kg1_triples_count": self.kg1_triples_count,
            "kg2_triples_count": self.kg2_triples_count,
            "all_attribute_triples_count": self.all_attribute_triples_count,
            "kg1_attribute_triples_count": self.kg1_attribute_triples_count,
            "kg2_attribute_triples_count": self.kg2_attribute_triples_count,
            "alignment_seeds_count": self.alignment_seeds_count,
            "valid_alignment_seeds_count": self.valid_alignment_seeds_count,
            "test_alignment_seeds_count": self.test_alignment_seeds_count,
            "train_alignment_seeds_count": self.train_alignment_seeds_count,
            "all_attribute_names_count": self.all_attribute_names_count,
            "all_attribute_values_count": self.all_attribute_values_count,
            "kg1_entities_count": self.kg1_entities_count,
            "kg2_entities_count": self.kg2_entities_count,
            "kg1_relations_count": self.kg1_relations_count,
            "kg2_relations_count": self.kg2_relations_count,
            "kg1_attribute_names_count": self.kg1_attribute_names_count,
            "kg2_attribute_names_count": self.kg2_attribute_names_count,
            "kg1_attribute_values_count": self.kg1_attribute_values_count,
            "kg2_attribute_values_count": self.kg2_attribute_values_count,
        }

    def dump(self) -> List[str]:
        dump = [
            "",
            "----------Metadata Info for Dataset:%s----------------" % self.dataset.name,
            "Total Entities           :%s" % self.entity_count,
            "Total Relations          :%s" % self.relation_count,
            "Total Attribute Names    :%s" % self.all_attribute_names_count,
            "Total Attribute Values   :%s" % self.all_attribute_values_count,

            "Total Triples            :%s" % self.triple_count,
            "Total Training Triples   :%s" % self.train_triples_count,
            "Total Testing Triples    :%s" % self.test_triples_count,
            "Total Validation Triples :%s" % self.valid_triples_count,

            "Total Attribute Triples  :%s" % self.all_attribute_triples_count,

            "Total Alignment Seeds              :%s" % self.alignment_seeds_count,
            "Total Validation Alignment Seeds   :%s" % self.valid_alignment_seeds_count,
            "Total Testing Alignment Seeds      :%s" % self.test_alignment_seeds_count,
            "Total Training Alignment Seeds     :%s" % self.train_alignment_seeds_count,

            "KG1",
            "triples           :%d" % self.kg1_triples_count,
            "attribute triples :%d" % self.kg1_attribute_triples_count,
            "entities          :%d" % self.kg1_entities_count,
            "relations         :%d" % self.kg1_relations_count,
            "attribute_names   :%d" % self.kg1_attribute_names_count,
            "attribute_values  :%d" % self.kg1_attribute_values_count,

            "KG2",
            "triples           :%d" % self.kg2_triples_count,
            "attribute triples :%d" % self.kg2_attribute_triples_count,
            "entities          :%d" % self.kg2_entities_count,
            "relations         :%d" % self.kg2_relations_count,
            "attribute_names   :%d" % self.kg2_attribute_names_count,
            "attribute_values  :%d" % self.kg2_attribute_values_count,
            "---------------------------------------------",
            "",
        ]
        return dump
