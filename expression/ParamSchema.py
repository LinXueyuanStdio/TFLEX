"""
@date: 2022/3/1
@description: null
"""
from inspect import signature
from typing import List, Tuple, Optional

from expression.symbol import Procedure

type_entity = "e"
type_relation = "r"
type_timestamp = "t"
NamedSample = List[Tuple[str, int]]


def is_entity(name) -> bool:
    return name.startswith(type_entity)


def is_relation(name) -> bool:
    return name.startswith(type_relation)


def is_timestamp(name) -> bool:
    return name.startswith(type_timestamp)


class FixedQuery:
    """
    推理过程的中间状态
    """

    def __init__(self, answers=None, timestamps=None, is_anchor=False):
        self.answers = answers if answers is not None else set()
        self.timestamps = timestamps if timestamps is not None else set()
        self.is_anchor = is_anchor

    def __len__(self):
        answers_len = len(self.answers) if self.answers is not None else 0
        timestamps_len = len(self.timestamps) if self.timestamps is not None else 0
        return answers_len + timestamps_len

    def __repr__(self):
        return f"answers={self.answers}, timestamps={self.timestamps}, is_anchor={self.is_anchor}"

    def from_tuple(self, t: Tuple[str, int]):
        self.is_anchor = True
        type_of_idx, idx = t
        if is_timestamp(type_of_idx):
            self.timestamps = {idx}
        else:
            self.answers = {idx}
        return self


class Placeholder:
    """
    占位符：anchor node
    计算图的起点
    采样时会自动往本占位符里填写采样到的 id
    """

    def __init__(self, name):
        self.name = name
        self.idx: Optional[int] = None

    def __repr__(self):
        return f"Placeholder({self.name}, idx={self.idx})"

    def clear(self):
        self.idx = None

    def fill(self, idx: int):
        self.idx = idx

    def fill_to_fixed_query(self, idx: int):
        self.idx = idx
        return self.to_fixed_query()

    def from_tuple(self, t: Tuple[str, int]):
        type_of_idx, idx = t
        self.name = type_of_idx
        self.idx = idx

    def to_tuple(self) -> Tuple[str, int]:
        return self.name, self.idx

    def to_fixed_query(self) -> FixedQuery:
        if is_timestamp(self.name):
            return FixedQuery(timestamps={self.idx}, is_anchor=True)
        else:
            return FixedQuery(answers={self.idx}, is_anchor=True)

    def fill_to(self, fixed_query: FixedQuery):
        if is_timestamp(self.name):
            fixed_query.timestamps = {self.idx}
        else:
            fixed_query.answers = {self.idx}


def get_param_name_list(func) -> List[str]:
    """
    根据函数签名，获得函数的入参列表
    """
    if isinstance(func, Procedure):
        return func.argnames
    sig_func = signature(func)
    return list(sig_func.parameters.keys())


def get_placeholder_list(func) -> List[Placeholder]:
    """
    从函数签名中生成占位符列表
    """
    params = get_param_name_list(func)
    return [Placeholder(name) for name in params]


def clear_placeholder_list(placeholder_list: List[Placeholder]):
    for placeholder in placeholder_list:
        placeholder.clear()


def placeholder_to_fixed_query(placeholder_list: List[Placeholder], fixed_query_list: List[FixedQuery]):
    for placeholder, fixed_query in zip(placeholder_list, fixed_query_list):
        placeholder.fill_to(fixed_query)


def placeholder2sample(placeholder_list: List[Placeholder]) -> List[int]:
    """
    将占位符中采样到的idx 转化为 用于保存的格式
    """
    return [i.idx for i in placeholder_list]


def placeholder2fixed(placeholder_list: List[Placeholder]) -> List[FixedQuery]:
    """
    将占位符中采样到的idx 转化为 用于保存的格式
    """
    return [i.to_fixed_query() for i in placeholder_list]


def sample2namedSample(func, sample: List[int]) -> NamedSample:
    params = get_param_name_list(func)
    return [(name, sample_id) for name, sample_id in zip(params, sample)]
