"""
@date: 2022/2/21
@description: null
"""
import random
from math import pi
from typing import List, Union

from .ParamSchema import Placeholder, FixedQuery, placeholder2fixed, get_param_name_list
from .symbol import Interpreter

query_structures = {
    # 1. 1-hop Pe and Pt, manually
    # 2. entity multi-hop
    "Pe2": "def Pe2(e1, r1, t1, r2, t2): return Pe(Pe(e1, r1, t1), r2, t2)",  # 2p
    "Pe3": "def Pe3(e1, r1, t1, r2, t2, r3, t3): return Pe(Pe(Pe(e1, r1, t1), r2, t2), r3, t3)",  # 3p
    # 3. time multi-hop
    "Pt_lPe": "def Pt_lPe(e1, r1, t1, r2, e2): return Pt(Pe(e1, r1, t1), r2, e2)",  # l for left (as head entity)
    "Pt_rPe": "def Pt_rPe(e1, r1, e2, r2, t1): return Pt(e1, r1, Pe(e2, r2, t1))",  # r for right (as tail entity)
    "Pe_Pt": "def Pe_Pt(e1, r1, e2, r2, e3): return Pe(e1, r1, Pt(e2, r2, e3))",  # at
    "Pe_aPt": "def Pe_aPt(e1, r1, e2, r2, e3): return Pe(e1, r1, after(Pt(e2, r2, e3)))",  # a for after
    "Pe_bPt": "def Pe_bPt(e1, r1, e2, r2, e3): return Pe(e1, r1, before(Pt(e2, r2, e3)))",  # b for before
    "Pe_nPt": "def Pe_nPt(e1, r1, e2, r2, e3): return Pe(e1, r1, next(Pt(e2, r2, e3)))",  # n for next
    # 4. entity and & time and
    "e2i": "def e2i(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2i
    "e3i": "def e3i(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Pe(e3, r3, t3))",  # 3i
    "t2i": "def t2i(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), Pt(e3, r2, e4))",  # t-2i
    "t3i": "def t3i(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), Pt(e5, r3, e6))",  # t-3i
    # 5. complex time and
    "e2i_Pe": "def e2i_Pe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Pe(e2, r3, t3))",  # pi
    "Pe_e2i": "def Pe_e2i(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(e2i(e1, r1, t1, e2, r2, t2), r3, t3)",  # ip
    "Pt_le2i": "def Pt_le2i(e1, r1, t1, e2, r2, t2, r3, e3): return Pt(e2i(e1, r1, t1, e2, r2, t2), r3, e3)",  # mix ip
    "Pt_re2i": "def Pt_re2i(e1, r1, e2, r2, t1, e3, r3, t2): return Pt(e1, r1, e2i(e2, r2, t1, e3, r3, t2))",  # mix ip
    "t2i_Pe": "def t2i_Pe(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), Pt(e3, r3, e4))",  # t-pi
    "Pe_t2i": "def Pe_t2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, t2i(e2, r2, e3, e4, r3, e5))",  # t-ip
    "Pe_at2i": "def Pe_at2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, after(t2i(e2, r2, e3, e4, r3, e5)))",
    "Pe_bt2i": "def Pe_bt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, before(t2i(e2, r2, e3, e4, r3, e5)))",
    "Pe_nt2i": "def Pe_nt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, next(t2i(e2, r2, e3, e4, r3, e5)))",
    "between": "def between(e1, r1, e2, e3, r2, e4): return TimeAnd(after(Pt(e1, r1, e2)), before(Pt(e3, r2, e4)))",  # between(t1, t2) == after t1 and before t2
    # 5. entity not
    "e2i_NPe": "def e2i_NPe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Not(Pe(Pe(e1, r1, t1), r2, t2)), Pe(e2, r3, t3))",  # pni = e2i_N(Pe(e1, r1, t1), r2, t2, e2, r3, t3)
    "e2i_PeN": "def e2i_PeN(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Not(Pe(e2, r3, t3)))",  # pin
    "Pe_e2i_Pe_NPe": "def Pe_e2i_Pe_NPe(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2))), r3, t3)",  # inp
    "e2i_N": "def e2i_N(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2)))",  # 2in
    "e3i_N": "def e3i_N(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Not(Pe(e3, r3, t3)))",  # 3in
    # 6. time not
    "t2i_NPt": "def t2i_NPt(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(TimeNot(Pt(Pe(e1, r1, t1), r2, e2)), Pt(e3, r3, e4))",  # t-pni
    "t2i_PtN": "def t2i_PtN(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), TimeNot(Pt(e3, r3, e4)))",  # t-pin
    "Pe_t2i_PtPe_NPt": "def Pe_t2i_PtPe_NPt(e1, r1, e2, r2, t2, r3, e3, e4, r4, e5): return Pe(e1, r1, TimeAnd(Pt(Pe(e2, r2, t2), r3, e3), TimeNot(Pt(e4, r4, e5))))",  # t-inp
    "t2i_N": "def t2i_N(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), TimeNot(Pt(e3, r2, e4)))",  # t-2in
    "t3i_N": "def t3i_N(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), TimeNot(Pt(e5, r3, e6)))",  # t-3in
    # 7. entity union & time union
    "e2u": "def e2u(e1, r1, t1, e2, r2, t2): return Or(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2u
    "Pe_e2u": "def Pe_e2u(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Or(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)",  # up
    "t2u": "def t2u(e1, r1, e2, e3, r2, e4): return TimeOr(Pt(e1, r1, e2), Pt(e3, r2, e4))",  # t-2u
    "Pe_t2u": "def Pe_t2u(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, TimeOr(Pt(e2, r2, e3), Pt(e4, r3, e5)))",  # t-up
    # 8. union-DM
    "e2u_DM": "def e2u_DM(e1, r1, t1, e2, r2, t2): return Not(And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2))))",  # 2u-DM
    "Pe_e2u_DM": "def Pe_e2u_DM(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Not(And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2)))), r3, t3)",  # up-DM
    "t2u_DM": "def t2u_DM(e1, r1, e2, e3, r2, e4): return TimeNot(TimeAnd(TimeNot(Pt(e1, r1, e2)), TimeNot(Pt(e3, r2, e4))))",  # t-2u-DM
    "Pe_t2u_DM": "def Pe_t2u_DM(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, TimeNot(TimeAnd(TimeNot(Pt(e2, r2, e3)), TimeNot(Pt(e4, r3, e5)))))",  # t-up-DM
    # 9. union-DNF
    "e2u_DNF": "def e2u_DNF(e1, r1, t1, e2, r2, t2): return Pe(e1, r1, t1), Pe(e2, r2, t2)",  # 2u_DNF
    "Pe_e2u_DNF": "def Pe_e2u_DNF(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Pe(e1, r1, t1), r3, t3), Pe(Pe(e2, r2, t2), r3, t3)",  # up_DNF
    "t2u_DNF": "def t2u_DNF(e1, r1, e2, e3, r2, e4): return Pt(e1, r1, e2), Pt(e3, r2, e4)",  # t-2u_DNF
    "Pe_t2u_DNF": "def Pe_t2u_DNF(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, Pt(e2, r2, e3)), Pe(e1, r1, Pt(e4, r3, e5))",  # t-up_DNF
}
union_query_structures = [
    "e2u", "Pe_e2u",  # 2u, up
    "t2u", "Pe_t2u",  # t-2u, t-up
]
train_query_structures = [
    # entity
    "Pe", "Pe2", "Pe3", "e2i", "e3i",  # 1p, 2p, 3p, 2i, 3i
    "e2i_NPe", "e2i_PeN", "Pe_e2i_Pe_NPe", "e2i_N", "e3i_N",  # npi, pni, inp, 2in, 3in
    # time
    "Pt", "Pt_lPe", "Pt_rPe", "Pe_Pt", "Pe_aPt", "Pe_bPt", "Pe_nPt",  # t-1p, t-2p
    "t2i", "t3i", "Pt_le2i", "Pt_re2i", "Pe_t2i", "Pe_at2i", "Pe_bt2i", "Pe_nt2i", "between",  # t-2i, t-3i
    "t2i_NPt", "t2i_PtN", "Pe_t2i_PtPe_NPt", "t2i_N", "t3i_N",  # t-npi, t-pni, t-inp, t-2in, t-3in
]
test_query_structures = train_query_structures + [
    # entity
    "e2i_Pe", "Pe_e2i",  # pi, ip
    "e2u", "Pe_e2u",  # 2u, up
    # time
    "t2i_Pe", "Pe_t2i",  # t-pi, t-ip
    "t2u", "Pe_t2u",  # t-2u, t-up
    # union-DM
    "e2u_DM", "Pe_e2u_DM",  # 2u-DM, up-DM
    "t2u_DM", "Pe_t2u_DM",  # t-2u-DM, t-up-DM
]


def is_to_predict_entity_set(query_name) -> bool:
    return query_name.startswith("e") or query_name.startswith("Pe")


def query_contains_union_and_we_should_use_DNF(query_name) -> bool:
    return query_name in union_query_structures


# def Pe_bt2i(e1, r1, e2, r2, e3, e4, r3, e5):
#     if t1 in before(t2):
#         return t2i(e2, r2, e3, e4, r3, e5)
#     else:
#         return t2i(e2, r2, e3, e4, r3, e5)

class BasicParser(Interpreter):
    """
    abstract class
    """

    def __init__(self, variables, neural_ops):
        alias = {
            "Pe": neural_ops["EntityProjection"],
            "Pt": neural_ops["TimeProjection"],
            "before": neural_ops["TimeBefore"],
            "after": neural_ops["TimeAfter"],
            "next": neural_ops["TimeNext"],
        }
        predefine = {
            "pi": pi,
        }
        functions = dict(**neural_ops, **alias, **predefine)
        super().__init__(usersyms=dict(**variables, **functions))
        self.func_cache = {}
        for _, qs in query_structures.items():
            self.eval(qs)

    def fast_function(self, func_name):
        if func_name in self.func_cache:
            return self.func_cache[func_name]
        func = self.eval(func_name)
        self.func_cache[func_name] = func
        return func

    def fast_args(self, func_name) -> List[str]:
        return get_param_name_list(self.fast_function(func_name))


class SamplingParser(BasicParser):
    def __init__(self, entity_ids: List[int], relation_ids: List[int], timestamp_ids: List[int],
                 sro_t, sor_t, srt_o, str_o, ors_t, trs_o, tro_s, rst_o, rso_t, t_sro, o_srt
                 ):
        # example
        # qe = Pe(e,r,after(Pt(e,r,e)))
        # [eid,rid,eid,rid,eid] = qe(e,r,e,r,e)
        # answers = qe(eid,rid,eid,rid,eid)
        # embedding = qe(eid,rid,eid,rid,eid)
        all_entity_ids = set(entity_ids)
        all_timestamp_ids = set(timestamp_ids)
        max_timestamp_id = max(timestamp_ids)

        variables = {
            "e": Placeholder("e"),
            "r": Placeholder("r"),
            "t": Placeholder("t"),
        }
        for e_id in entity_ids:
            variables[f"e{e_id}"] = FixedQuery(answers={e_id}, is_anchor=True)
        for r_id in relation_ids:
            variables[f"r{r_id}"] = FixedQuery(answers={r_id}, is_anchor=True)
        for t_id in timestamp_ids:
            variables[f"t{t_id}"] = FixedQuery(timestamps={t_id}, is_anchor=True)

        def find_entity(s: Union[FixedQuery, Placeholder], r: Union[FixedQuery, Placeholder], t: Union[FixedQuery, Placeholder]):
            s_is_missing, r_is_missing, t_is_missing = isinstance(s, Placeholder), isinstance(r, Placeholder), isinstance(t, Placeholder)
            if s_is_missing and r_is_missing and t_is_missing:
                si = random.choice(list(srt_o.keys()))
                s = s.fill_to_fixed_query(si)

                rj = random.choice(list(srt_o[si].keys()))
                r = r.fill_to_fixed_query(rj)

                tk = random.choice(list(srt_o[si][rj].keys()))
                t = t.fill_to_fixed_query(tk)
            elif not s_is_missing and r_is_missing and t_is_missing:
                choices = list(s.answers)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(srt_o[si].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(srt_o[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif s_is_missing and not r_is_missing and t_is_missing:
                choices = list(r.answers)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(rst_o[rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)

                choices = list(rst_o[rj][si].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)
            elif s_is_missing and r_is_missing and not t_is_missing:
                choices = list(t.timestamps)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(trs_o[tk].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(trs_o[tk][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif s_is_missing and not r_is_missing and not t_is_missing:
                choices = list(t.timestamps)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(r.answers)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(trs_o[tk][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif not s_is_missing and r_is_missing and not t_is_missing:
                choices = list(s.answers)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(t.timestamps)
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)

                choices = list(str_o[si][tk].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)
            elif not s_is_missing and not r_is_missing and t_is_missing:
                choices = list(s.answers)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(r.answers)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(srt_o[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                tk = random.choice(choices)
                t = t.fill_to_fixed_query(tk)

            answers = set()
            for si in s.answers:
                for rj in r.answers:
                    for tk in t.timestamps:
                        answers = answers | set(srt_o[si][rj][tk])
            # print("find_entity", answers)
            return answers

        def find_timestamp(s: Union[FixedQuery, Placeholder], r: Union[FixedQuery, Placeholder], o: Union[FixedQuery, Placeholder]):
            s_is_missing, r_is_missing, o_is_missing = isinstance(s, Placeholder), isinstance(r, Placeholder), isinstance(o, Placeholder)
            if s_is_missing and r_is_missing and o_is_missing:
                si = random.choice(list(sro_t.keys()))
                s = s.fill_to_fixed_query(si)

                rj = random.choice(list(sro_t[si].keys()))
                r = r.fill_to_fixed_query(rj)

                ok = random.choice(list(sro_t[si][rj].keys()))
                o = o.fill_to_fixed_query(ok)
            elif not s_is_missing and r_is_missing and o_is_missing:
                choices = list(s.answers)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(sro_t[si].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(sro_t[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)
                o = o.fill_to_fixed_query(ok)
            elif s_is_missing and not r_is_missing and o_is_missing:
                choices = list(r.answers)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(rso_t[rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)

                choices = list(rso_t[rj][si].keys())
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)
                o = o.fill_to_fixed_query(ok)
            elif s_is_missing and r_is_missing and not o_is_missing:
                choices = list(o.answers)
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)

                choices = list(ors_t[ok].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)

                choices = list(ors_t[ok][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif s_is_missing and not r_is_missing and not o_is_missing:
                choices = list(o.answers)
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)

                choices = list(r.answers)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(tro_s[ok][rj].keys())
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)
                s = s.fill_to_fixed_query(si)
            elif not s_is_missing and r_is_missing and not o_is_missing:
                choices = list(s.answers)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(o.answers)
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)

                choices = list(sor_t[si][ok].keys())
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)
                r = r.fill_to_fixed_query(rj)
            elif not s_is_missing and not r_is_missing and o_is_missing:
                choices = list(s.answers)
                if len(choices) <= 0:
                    return set()
                si = random.choice(choices)

                choices = list(r.answers)
                if len(choices) <= 0:
                    return set()
                rj = random.choice(choices)

                choices = list(sro_t[si][rj].keys())
                if len(choices) <= 0:
                    return set()
                ok = random.choice(choices)
                o = o.fill_to_fixed_query(ok)

            timestamps = set()
            for si in s.answers:
                for rj in r.answers:
                    for ok in o.answers:
                        timestamps = timestamps | set(sro_t[si][rj][ok])
            # print("find_timestamp", timestamps)
            return timestamps

        neural_ops = {  # 4+4+3
            "And": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "And3": lambda q1, q2, q3: FixedQuery(answers=q1.answers & q2.answers & q3.answers, timestamps=q1.timestamps & q2.timestamps & q3.timestamps),
            "Or": lambda q1, q2: FixedQuery(answers=q1.answers | q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "Not": lambda x: FixedQuery(answers=all_entity_ids - x.answers, timestamps=x.timestamps),
            "EntityProjection": lambda e1, r1, t1: FixedQuery(answers=find_entity(e1, r1, t1)),
            "TimeProjection": lambda e1, r1, e2: FixedQuery(timestamps=find_timestamp(e1, r1, e2)),
            "TimeAnd": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "TimeAnd3": lambda q1, q2, q3: FixedQuery(answers=q1.answers & q2.answers & q3.answers, timestamps=q1.timestamps & q2.timestamps & q3.timestamps),
            "TimeOr": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps | q2.timestamps),
            "TimeNot": lambda x: FixedQuery(answers=x.answers, timestamps=all_timestamp_ids - x.timestamps if len(x.timestamps) > 0 else all_timestamp_ids),
            "TimeBefore": lambda x: FixedQuery(answers=x.answers, timestamps=set([t for t in timestamp_ids if t < min(x.timestamps)] if len(x.timestamps) > 0 else all_timestamp_ids)),
            "TimeAfter": lambda x: FixedQuery(answers=x.answers, timestamps=set([t for t in timestamp_ids if t > max(x.timestamps)] if len(x.timestamps) > 0 else all_timestamp_ids)),
            "TimeNext": lambda x: FixedQuery(answers=x.answers, timestamps=set([min(t + 1, max_timestamp_id) for t in x.timestamps] if len(x.timestamps) > 0 else all_timestamp_ids)),
        }

        # fast sampling
        valid_e2i_o_list = [k for k, v in o_srt.items() if len(v) >= 2]

        def fast_e2i_targeted(e1, r1, t1, e2, r2, t2, target: int):
            (e1_idx, r1_idx, t1_idx), (e2_idx, r2_idx, t2_idx) = tuple(random.sample(list(o_srt[target]), k=2))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            t1.fill(t1_idx)
            e2.fill(e2_idx)
            r2.fill(r2_idx)
            t2.fill(t2_idx)
            placeholders = [e1, r1, t1, e2, r2, t2]
            return self.fast_function("e2i")(*placeholder2fixed(placeholders))

        def fast_e2i(e1, r1, t1, e2, r2, t2):
            o_idx = random.choice(valid_e2i_o_list)
            return fast_e2i_targeted(e1, r1, t1, e2, r2, t2, target=o_idx)

        valid_e3i_o_list = [k for k, v in o_srt.items() if len(v) >= 3]

        def fast_e3i(e1, r1, t1, e2, r2, t2, e3, r3, t3):
            o = random.choice(valid_e3i_o_list)
            (e1_idx, r1_idx, t1_idx), (e2_idx, r2_idx, t2_idx), (e3_idx, r3_idx, t3_idx) = tuple(random.sample(list(o_srt[o]), k=3))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            t1.fill(t1_idx)
            e2.fill(e2_idx)
            r2.fill(r2_idx)
            t2.fill(t2_idx)
            e3.fill(e3_idx)
            r3.fill(r3_idx)
            t3.fill(t3_idx)
            placeholders = [e1, r1, t1, e2, r2, t2, e3, r3, t3]
            return self.fast_function("e3i")(*placeholder2fixed(placeholders))

        valid_t2i_t_list = [k for k, v in t_sro.items() if len(v) >= 2]

        def fast_t2i(e1, r1, e2, e3, r2, e4):
            t = random.choice(valid_t2i_t_list)
            (e1_idx, r1_idx, e2_idx), (e3_idx, r2_idx, e4_idx) = tuple(random.sample(list(t_sro[t]), k=2))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            e2.fill(e2_idx)
            e3.fill(e3_idx)
            r2.fill(r2_idx)
            e4.fill(e4_idx)
            placeholders = [e1, r1, e2, e3, r2, e4]
            return self.fast_function("t2i")(*placeholder2fixed(placeholders))

        valid_t3i_t_list = [k for k, v in t_sro.items() if len(v) >= 3]

        def fast_t3i(e1, r1, e2, e3, r2, e4, e5, r3, e6):
            t = random.choice(valid_t3i_t_list)
            (e1_idx, r1_idx, e2_idx), (e3_idx, r2_idx, e4_idx), (e5_idx, r3_idx, e6_idx) = tuple(random.sample(list(t_sro[t]), k=3))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            e2.fill(e2_idx)
            e3.fill(e3_idx)
            r2.fill(r2_idx)
            e4.fill(e4_idx)
            e5.fill(e5_idx)
            r3.fill(r3_idx)
            e6.fill(e6_idx)
            placeholders = [e1, r1, e2, e3, r2, e4, e5, r3, e6]
            return self.fast_function("t3i")(*placeholder2fixed(placeholders))

        def fast_Pt_le2i(e1, r1, t1, e2, r2, t2, r3, e3):
            o_idx = random.choice(list(set(valid_e2i_o_list) & set(sro_t.keys())))
            q = fast_e2i_targeted(e1, r1, t1, e2, r2, t2, target=o_idx)
            return self.fast_function("Pt")(q, r3, e3)

        def fast_Pt_re2i(e1, r1, e2, r2, t1, e3, r3, t2):
            o_idx = random.choice(list(set(valid_e2i_o_list) & set(ors_t.keys())))
            q = fast_e2i_targeted(e2, r2, t1, e3, r3, t2, target=o_idx)
            return self.fast_function("Pt")(e1, r1, q)

        def fast_Pe_targeted(e1, r1, t1, target: int):
            e1_idx, r1_idx, t1_idx = random.choice(list(o_srt[target]))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            t1.fill(t1_idx)
            return srt_o[e1_idx][r1_idx][t1_idx]

        def fast_Pt_targeted(e1, r1, e2, target: int):
            e1_idx, r1_idx, e2_idx = random.choice(list(t_sro[target]))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            e2.fill(e2_idx)
            return sro_t[e1_idx][r1_idx][e2_idx]

        def fast_Pt_lPe_targeted(e1, r1, t1, r2, e2, target: int):
            # Pt(Pe(e1, r1, t1), r2, e2)
            e1_idx, r1_idx, e2_idx = random.choice(list(t_sro[target]))
            e1_ids = fast_Pe_targeted(e1, r1, t1, target=e1_idx)
            r2.fill(r1_idx)
            e2.fill(e2_idx)
            answers = set()
            for idx in e1_ids:
                answers = answers | sro_t[idx][r1_idx][e2_idx]
            return answers

        def fast_Pe2_targeted(e1, r1, t1, r2, t2, target: int):
            # Pe(Pe(e1, r1, t1), r2, t2)
            e1_idx, r2_idx, t2_idx = random.choice(list(o_srt[target]))
            e1_ids = fast_Pe_targeted(e1, r1, t1, target=e1_idx)
            r2.fill(r2_idx)
            t2.fill(t2_idx)
            answers = set()
            for idx in e1_ids:
                answers = answers | srt_o[idx][r2_idx][t2_idx]
            return answers

        def fast_Pt_lPe(e1, r1, t1, r2, e2):
            # return Pt(Pe(e1, r1, t1), r2, e2)
            t = random.choice(list(t_sro.keys()))
            t_ids = fast_Pt_lPe_targeted(e1, r1, t1, r2, e2, target=t)
            return FixedQuery(timestamps=t_ids)

        def fast_t2i_NPt(e1, r1, t1, r2, e2, e3, r3, e4):
            # return TimeAnd(TimeNot(Pt(Pe(e1, r1, t1), r2, e2)), Pt(e3, r3, e4))
            t_choices = list(t_sro.keys())
            t = random.choice(t_choices)
            choices = set(all_timestamp_ids - {t}) & set(t_choices)
            while len(choices) <= 0:
                t = random.choice(t_choices)
                choices = set(all_timestamp_ids - {t}) & set(t_choices)
            not_t = random.choice(list(choices))
            right_t_ids = fast_Pt_targeted(e3, r3, e4, target=t)
            left_t_ids = all_timestamp_ids - fast_Pt_lPe_targeted(e1, r1, t1, r2, e2, target=not_t)
            return FixedQuery(timestamps=left_t_ids & right_t_ids)

        def fast_e2i_NPe(e1, r1, t1, r2, t2, e2, r3, t3):
            # return And(Not(Pe(Pe(e1, r1, t1), r2, t2)), Pe(e2, r3, t3))
            o_choices = list(o_srt.keys())
            o = random.choice(o_choices)
            choices = set(all_entity_ids - {o}) & set(o_choices)
            while len(choices) <= 0:
                o = random.choice(list(o_srt.keys()))
                choices = set(all_entity_ids - {o}) & set(o_choices)
            not_o = random.choice(list(choices))
            right_o_ids = fast_Pe_targeted(e2, r3, t3, target=o)
            left_o_ids = all_entity_ids - fast_Pe2_targeted(e1, r1, t1, r2, t2, target=not_o)
            return FixedQuery(answers=left_o_ids & right_o_ids)

        def fast_Pe_Pt(e1, r1, e2, r2, e3):
            # return Pe(e1, r1, Pt(e2, r2, e3))
            o_idx = random.choice(list(o_srt.keys()))
            e1_idx, r1_idx, t1_idx = random.choice(list(o_srt[o_idx]))
            e1.fill(e1_idx)
            r1.fill(r1_idx)
            o_ids = set()
            t_ids = fast_Pt_targeted(e2, r2, e3, target=t1_idx)
            for t_idx in t_ids:
                o_ids = o_ids | srt_o[e1_idx][r1_idx][t_idx]
            return FixedQuery(answers=o_ids)

        def fast_Pe_e2i(e1, r1, t1, e2, r2, t2, r3, t3):
            # return Pe(And(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)
            o_idx = random.choice(list(set(valid_e2i_o_list) & set(sro_t.keys())))
            q = fast_e2i_targeted(e1, r1, t1, e2, r2, t2, target=o_idx)
            return self.fast_function("Pe")(q, r3, t3)

        self.fast_ops = {
            "fast_e2i": fast_e2i,
            "fast_e3i": fast_e3i,
            "fast_t2i": fast_t2i,
            "fast_t3i": fast_t3i,
            "fast_Pt_le2i": fast_Pt_le2i,
            "fast_Pt_re2i": fast_Pt_re2i,
            "fast_Pt_lPe": fast_Pt_lPe,
            "fast_t2i_NPt": fast_t2i_NPt,
            "fast_e2i_NPe": fast_e2i_NPe,
            "fast_Pe_Pt": fast_Pe_Pt,
            "fast_Pe_e2i": fast_Pe_e2i,
        }
        # Pe_t2i_PtPe_NPt 在 train 中有答案，在 test 中没有答案
        # 已解决：test中无答案则重新抽取
        super().__init__(variables=variables, neural_ops=dict(**neural_ops, **self.fast_ops))


class NeuralParser(BasicParser):
    def __init__(self, neural_ops, variables=None):
        if variables is None:
            variables = {}
        must_implement_neural_ops = [
            "And",
            "And3",
            "Or",
            "Not",
            "EntityProjection",
            "TimeProjection",
            "TimeAnd",
            "TimeAnd3",
            "TimeOr",
            "TimeNot",
            "TimeBefore",
            "TimeAfter",
            "TimeNext",
        ]
        for op in must_implement_neural_ops:
            if op not in neural_ops:
                raise Exception(f"{op} Not Found! You MUST implement neural operation '{op}'")
        super().__init__(variables=variables, neural_ops=neural_ops)
