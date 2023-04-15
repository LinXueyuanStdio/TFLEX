"""
@date: 2022/3/1
@description: null
"""
from inspect import signature
from typing import List, Set, Tuple, Optional, Union

from expression.symbol import Procedure

NamedSample = List[Tuple[str, int]]


def is_entity(name) -> bool:
    return name.startswith("e") or name.startswith("s") or name.startswith("o")


def is_relation(name) -> bool:
    return name.startswith("r")


def is_timestamp(name) -> bool:
    return name.startswith("t")


class QuerySet:
    """
    推理过程的中间状态
    """

    def __init__(self, ids=None):
        self.ids = ids if ids is not None else set()

    def __len__(self):
        ids_len = len(self.ids) if self.ids is not None else 0
        return ids_len

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ids.__repr__()})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.ids.__repr__()})"

    def __contains__(a, b):
        if a.__class__.__name__ != b.__class__.__name__:
            return False
        if a is not QuerySet or b is not QuerySet:
            return False
        return a.ids.issuperset(b.ids)

    def __eq__(self, __value: object) -> bool:
        if __value is not QuerySet:
            return False
        if self.__class__.__name__ != __value.__class__.__name__:
            return False
        return self.ids == __value.ids

    def __ne__(self, __value: object) -> bool:
        if __value is not QuerySet:
            return True
        if self.__class__.__name__ != __value.__class__.__name__:
            return True
        return self.ids != __value.ids

    def __add__(a, b):
        if a.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for +: '{a.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = a.ids | b.ids
        return a.__class__(ids)

    def __minus__(a, b):
        if a.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for -: '{a.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = a.ids - b.ids
        return a.__class__(ids)

    def __and__(a, b):
        if a.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for &: '{a.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = a.ids & b.ids
        return a.__class__(ids)

    def __or__(a, b):
        if a.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for |: '{a.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = a.ids | b.ids
        return a.__class__(ids)

    def __xor__(a, b):
        if a.__class__.__name__ != b.__class__.__name__:
            raise TypeError(f"unsupported operand type(s) for ^: '{a.__class__.__name__}' and '{b.__class__.__name__}'")
        ids = a.ids ^ b.ids
        return a.__class__(ids)


class EntitySet(QuerySet):
    def __init__(self, entity: Union[int, Set]) -> None:
        if entity is int:
            entity = {entity}
        super().__init__(entity)


class TimeSet(QuerySet):
    def __init__(self, timestamp: Union[int, Set]) -> None:
        if timestamp is int:
            timestamp = {timestamp}
        super().__init__(timestamp)


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

    def to_fixed_query(self) -> QuerySet:
        if is_timestamp(self.name):
            return TimeSet({self.idx})
        else:
            return EntitySet({self.idx})

    def fill_to(self, fixed_query: QuerySet):
        fixed_query.ids = {self.idx}


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


def placeholder_to_fixed_query(placeholder_list: List[Placeholder], fixed_query_list: List[QuerySet]):
    for placeholder, fixed_query in zip(placeholder_list, fixed_query_list):
        placeholder.fill_to(fixed_query)


def placeholder2sample(placeholder_list: List[Placeholder]) -> List[int]:
    """
    将占位符中采样到的idx 转化为 用于保存的格式
    """
    return [i.idx for i in placeholder_list]


def placeholder2fixed(placeholder_list: List[Placeholder]) -> List[QuerySet]:
    """
    将占位符中采样到的idx 转化为 用于保存的格式
    """
    return [i.to_fixed_query() for i in placeholder_list]


def sample2namedSample(func, sample: List[int]) -> NamedSample:
    params = get_param_name_list(func)
    return [(name, sample_id) for name, sample_id in zip(params, sample)]
