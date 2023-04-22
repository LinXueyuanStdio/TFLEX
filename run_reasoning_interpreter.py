"""
@date: 2021/10/27
@description: 交互式表达式解释器，输入 query 输出 answer
"""
import cmd
import random
import sys
import traceback
from typing import List, Optional, Tuple

import click
import torch
import json

import expression
from ComplexTemporalQueryData import *
from expression.ParamSchema import EntitySet, TimeSet
from expression.TFLEX_DSL import BasicParser
from expression.symbol import Interpreter
from expression.FLEX_DSL import query_structures
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.ModelParamStore import ModelParamStoreSchema
from train_TCQE_TFLEX import TYPE_token
from toolbox.utils.Log import blue, cyan, green, orange, grey, yellow, red, bold_red, reset

class ExpressionInterpreter(cmd.Cmd):
    """
    Interactive command line interpreter that applies the expression line parser
    to the provided input.
    """

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = cyan + 'you: ' + reset
        variables, functions = self.interactive_ops()
        self.default_parser = Interpreter(usersyms=dict(**variables, **functions))
        self.parser: Interpreter = self.default_parser

        self.data: Optional[ComplexQueryData] = None
        self.dataset_name: Optional[str] = None
        self.dataset: Optional[str] = None

        self.neural_parser: Optional[expression.NeuralParser] = None
        self.groundtruth_parser: Optional[expression.SamplingParser] = None

    def interactive_ops(self):
        variables = {
            "data": None,
        }
        functions = {
            "commands": lambda: """
available commands:
    list_queries() : list all predefined queries
    use_dataset(data_home="./data", dataset_name="ICEWS14"):
        data_home: The folder path to dataset.
        dataset_name: Which dataset to use: ICEWS14, ICEWS05_15, GDELT.
        usage example:
            >>> use_dataset()
            >>> use_dataset("data")
            >>> use_dataset("data", "ICEWS14")
            >>> use_dataset(dataset_name="GDELT")
    list_entities(k=5): randomly list k entities, -1 to list all
    list_relations(k=5): randomly list k relations, -1 to list all
    list_timestamps(k=5): randomly list k timestamps, -1 to list all
    sample(task_name="Pe", k=5): randomly sample k entities, -1 to list all
    use_neural_interpreter(name="TFLEX"):
        alias = use_n()
        use neural interpreter to answer queries
        this function will load the trained TCQE model to answer queries.
    use_groundtruth_interpreter():
        alias = use_gt()
        use groundtruth interpreter to answer queries
        this interpreter will perform reasoning by subgraph matching over the temporal knowledge graph.
        the answer may be wrong if there exists missing link. Note that TKG is incomplete.

        tensor_id_of(idx: int) -> torch.LongTensor
        tensor_entity(entity: str) -> torch.LongTensor
        tensor_relation(relation: str) -> torch.LongTensor
        tensor_timestamp(timestamp: str) -> torch.LongTensor
        entity_token(entity: Union[int, str]) -> TYPE_token
        relation_token(relation: Union[int, str]) -> TYPE_token
        timestamp_token(timestamp: Union[int, str]) -> TYPE_token
        neural_answer_entities(query, topk=5):
            alias = ne(query, topk=5)
            use neural interpreter to answer query and return k entities
        neural_answer_timestamps(query, topk=5):
            alias = nt(query, topk=5)
            use neural interpreter to answer query and return k timestamps
    groundtruth_answer(query):
        alias = gt(query)
        use groundtruth interpreter to answer query and return k entities
    answer_entities(query, k=5):
        alias = e(query, k=5)
        auto use neural interpreter or groundtruth interpreter to answer query and return k entities
    answer_timestamps(query, k=5):
        alias = t(query, k=5)
        auto use neural interpreter or groundtruth interpreter to answer query and return k timestamps
    commands():
        show this help message
            """,
            "list_queries": lambda: json.dumps(query_structures, indent=2),
            "use_dataset": self.use_dataset,
            "list_entities": self.list_entities,
            "list_relations": self.list_relations,
            "list_timestamps": self.list_timestamps,
            "list_triples": self.list_triples,
            "list_triples_ids": self.list_triples_ids,
            "sample": self.sample,

            "use_neural_interpreter": self.use_neural_interpreter,
            "use_n": self.use_neural_interpreter,
            "neural_answer_entities": self.neural_answer_entities,
            "ne": self.neural_answer_entities,
            "neural_answer_timestamps": self.neural_answer_timestamps,
            "nt": self.neural_answer_timestamps,

            "tensor_id_of": self.tensor_id_of,
            "tensor_entity": self.tensor_entity,
            "tensor_relation": self.tensor_relation,
            "tensor_timestamp": self.tensor_timestamp,
            "entity_token": self.entity_token,
            "relation_token": self.relation_token,
            "timestamp_token": self.timestamp_token,

            "use_groundtruth_interpreter": self.use_groundtruth_interpreter,
            "use_gt": self.use_groundtruth_interpreter,
            "groundtruth_answer": self.groundtruth_answer,
            "gt": self.groundtruth_answer,

            "answer_entities": self.answer_entities,
            "e": self.answer_entities,
            "answer_timestamps": self.answer_timestamps,
            "t": self.answer_timestamps,
        }
        return variables, functions

    def use_dataset(self, data_home="data", dataset_name="ICEWS14"):
        if dataset_name == "ICEWS14":
            dataset = ICEWS14(data_home)
        elif dataset_name == "ICEWS05_15":
            dataset = ICEWS05_15(data_home)
        elif dataset_name == "GDELT":
            dataset = GDELT(data_home)
        cache = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
        data = ComplexQueryData(dataset, cache_path=cache)
        data.preprocess_data_if_needed()
        data.load_cache([
            "meta",
            "all_entities", "all_relations", "all_timestamps",
            "entities_ids", "relations_ids", "timestamps_ids",
        ])
        self.data = data
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.parser.symtable.update({
            "data": data,
            "dataset": dataset,
        })
        return f"using {dataset}"

    def list_entities(self, k=5):
        if self.data is None:
            return "you should load dataset first. please call `use_dataset()`"
        if k == -1:
            return self.data.all_entities
        return random.sample(self.data.all_entities, k)

    def list_relations(self, k=5):
        if self.data is None:
            return "you should load dataset first. please call `use_dataset()`"
        if k == -1:
            return self.data.all_relations
        return random.sample(self.data.all_relations, k)

    def list_timestamps(self, k=5):
        if self.data is None:
            return "you should load dataset first. please call `use_dataset()`"
        if k == -1:
            return self.data.all_timestamps
        return random.sample(self.data.all_timestamps, k)

    def list_triples(self, k=5):
        if self.data is None:
            return "you should load dataset first. please call `use_dataset()`"
        if self.data.all_triples is None or len(self.data.all_triples) <= 0:
            self.data.load_cache(["all_triples"])
        if k == -1:
            return self.data.all_triples
        return random.sample(self.data.all_triples, k)

    def list_triples_ids(self, k=5):
        if self.data is None:
            return "you should load dataset first. please call `use_dataset()`"
        if self.data.all_triples_ids is None or len(self.data.all_triples_ids) <= 0:
            self.data.load_cache(["all_triples_ids"])
        if k == -1:
            return self.data.all_triples_ids
        return random.sample(self.data.all_triples_ids, k)

    def sample(self, task_name: str = "Pe", k=3):
        task_data = self.data.load_cache_by_tasks([task_name], "test")[task_name]
        args = task_data["args"]
        args = ", ".join(args)
        qa = task_data["queries_answers"]
        sample_data = random.sample(qa, k)
        print(f"sample {k} {task_name}({args}) data:")
        for queries, easy_answer, hard_answer in sample_data:
            print(f"queries: {queries}, easy_answer: {easy_answer}, hard_answer: {hard_answer}")
        return sample_data

    def use_neural_interpreter(self, name,
                               hidden_dim=800, gamma=30.0, center_reg=0.0,
                               test_batch_size=100, input_dropout=0.2, device="cuda"):
        from train_TCQE_TFLEX import TFLEX
        if self.dataset is None or self.data is None:
            return "you should load dataset first. please call `use_dataset()`"
        output = OutputSchema(self.dataset.name + "-" + name)
        entity_count = self.data.entity_count
        relation_count = self.data.relation_count
        timestamp_count = self.data.timestamp_count
        max_relation_id = relation_count
        model = TFLEX(
            nentity=entity_count,
            nrelation=relation_count + max_relation_id,  # with reverse relations
            ntimestamp=timestamp_count,
            hidden_dim=hidden_dim,
            gamma=gamma,
            center_reg=center_reg,
            test_batch_size=test_batch_size,
            drop=input_dropout,
        )
        self.model = model.to(device)
        self.device = device
        self.model_param_store = ModelParamStoreSchema(output.pathSchema)
        start_step, _, best_score = self.model_param_store.load_best(model, None)
        print(f"load best model at step {start_step} with score {best_score}")
        self.neural_parser = self.model.parser
        self.switch_parser_to(self.neural_parser)
        return "using neural interpreter"

    def tensor_id_of(self, idx: int) -> torch.LongTensor:
        return torch.LongTensor([[idx]]).to(self.device)

    def tensor_entity(self, entity: str):
        return self.tensor_id_of(self.data.entities_ids[entity])

    def tensor_relation(self, relation: str):
        return self.tensor_id_of(self.data.relations_ids[relation])

    def tensor_timestamp(self, timestamp: str):
        return self.tensor_id_of(self.data.timestamps_ids[timestamp])

    def entity_token(self, entity: Union[int, str]) -> TYPE_token:
        if isinstance(entity, int):
            idx = self.tensor_id_of(entity)
        elif isinstance(entity, str):
            idx = self.tensor_entity(entity)
        else:
            raise TypeError(f"entity should be int or str, but got {type(entity)}")
        return self.model.entity_token(idx)

    def relation_token(self, relation: Union[int, str]) -> TYPE_token:
        if isinstance(relation, int):
            idx = self.tensor_id_of(relation)
        elif isinstance(relation, str):
            idx = self.tensor_relation(relation)
        else:
            raise TypeError(f"relation should be int or str, but got {type(relation)}")
        return self.model.relation_token(idx)

    def timestamp_token(self, timestamp: Union[int, str]) -> TYPE_token:
        if isinstance(timestamp, int):
            idx = self.tensor_id_of(timestamp)
        elif isinstance(timestamp, str):
            idx = self.tensor_timestamp(timestamp)
        else:
            raise TypeError(f"timestamp should be int or str, but got {type(timestamp)}")
        return self.model.timestamp_token(idx)

    def neural_answer_entities(self, query_token: TYPE_token, topk=10):
        answer_range = self.data.entity_count
        candidate_answer = torch.LongTensor(range(answer_range)).to(self.device)
        return self.neural_answer(query_token, candidate_answer, predict_entity=True, topk=topk)

    def neural_answer_timestamps(self, query_token: TYPE_token, topk=10):
        answer_range = self.data.timestamp_count
        candidate_answer = torch.LongTensor(range(answer_range)).to(self.device)
        return self.neural_answer(query_token, candidate_answer, predict_entity=False, topk=topk)

    def neural_answer(self, predict: TYPE_token, answer: torch.LongTensor, predict_entity: bool, topk=10):
        # self.forward_predict(query_structure, query, answer)
        # self.model.single_predict(query_structure, query_tensor)
        all_predict: TYPE_token = tuple([i.unsqueeze(dim=1) for i in predict])  # (B, 1, d)
        scores = self.model.scoring_to_answers(answer, all_predict, predict_entity=predict_entity, DNF_predict=False)
        value, idxs = torch.topk(scores, k=topk, dim=1)
        answers = []
        for i in range(len(idxs)):
            row = []
            for j in range(len(idxs[i])):
                idx = idxs[i][j].item()
                score = value[i][j].item()
                if predict_entity:
                    row += [(self.data.all_entities[idx], score)]
                else:
                    row += [(self.data.all_timestamps[idx], score)]
            answers.append(row)
        return answers

    def use_groundtruth_interpreter(self):
        self.data.load_cache([
            "train_triples_ids", "valid_triples_ids", "test_triples_ids",
        ])
        max_relation_id = self.data.relation_count
        relations_ids_with_reverse = self.data.relations_ids + [r + max_relation_id for r in self.data.relations_ids]

        def append_reverse(triples):
            nonlocal max_relation_id
            res = []
            for s, r, o, t in triples:
                res.append((s, r, o, t))
                res.append((o, r + max_relation_id, s, t))
            return res

        train_triples_ids = append_reverse(self.data.train_triples_ids)
        valid_triples_ids = append_reverse(self.data.valid_triples_ids)
        test_triples_ids = append_reverse(self.data.test_triples_ids)
        test_sro_t, test_sor_t, test_srt_o, test_str_o, \
            test_ors_t, test_trs_o, test_tro_s, test_rst_o, \
            test_rso_t, test_t_sro, test_o_srt = build_mapping_simple(train_triples_ids + valid_triples_ids + test_triples_ids)
        test_parser = expression.SamplingParser(
            self.data.entities_ids, relations_ids_with_reverse, self.data.timestamps_ids, test_sro_t, test_sor_t,
            test_srt_o, test_str_o, test_ors_t, test_trs_o, test_tro_s, test_rst_o, test_rso_t, test_t_sro, test_o_srt)
        self.groundtruth_parser = test_parser
        self.switch_parser_to(self.groundtruth_parser)
        return "using groundtruth interpreter"

    def groundtruth_answer(self, query: Union[EntitySet, QuerySet, TimeSet], topk=10):
        if type(query) is EntitySet:
            return [self.data.all_entities[idx] for idx in query.ids]
        elif type(query) is TimeSet:
            return [self.data.all_timestamps[idx] for idx in query.ids]
        else:
            return [self.data.all_relations[idx] for idx in query.ids]
        # answers = random.sample(answers, min(topk, len(answers)))
        # timestamps = random.sample(timestamps, min(topk, len(timestamps)))

    def answer_entities(self, query, topk=10):
        if self.parser == self.neural_parser:
            return self.neural_answer_entities(query, topk)
        return self.groundtruth_answer(query, topk)

    def answer_timestamps(self, query, topk=10):
        if self.parser == self.neural_parser:
            return self.neural_answer_timestamps(query, topk)
        return self.groundtruth_answer(query, topk)

    def switch_parser_to(self, parser: BasicParser):
        if self.parser == parser:
            return
        parser.symtable.update(self.parser.symtable)
        self.parser = parser

    def switch_parser(self, mode="neural"):
        if mode == "neural":
            if self.neural_parser is None:
                return "you should load neural parser first. please call `use_neural_interpreter()`"
            self.switch_parser_to(self.neural_parser)
        elif mode == "groundtruth":
            if self.groundtruth_parser is None:
                return "you should load groundtruth parser first. please call `use_groundtruth_interpreter()`"
            self.switch_parser_to(self.groundtruth_parser)
        else:
            self.switch_parser_to(self.default_parser)
            if mode != "default":
                return f"unknown parser mode {mode}. reset to default parser"

    def default(self, line):
        try:
            output = self.parser.eval(line)
            if output is not None:
                self.stdout.write(f"{yellow}bot: {reset}{green}{output}{reset}\n")

        except SyntaxError:
            traceback.print_exc(0)

    def do_quit(self, line):
        """
        Exit the interpreter.
        """

        if line != '' and line != '()':
            self.stdout.write(line + '\n')
        self._quit()

    @staticmethod
    def _quit():
        sys.exit(1)


def main():
    ExpressionInterpreter().cmdloop()


if __name__ == '__main__':
    main()
