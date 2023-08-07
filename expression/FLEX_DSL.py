"""
@date: 2022/2/21
@description: null
"""
query_structures = {
    # 1. 1-hop relational projection 'P' is predefined, so as And, Not, Or, And3
    # "P1": "def P1(e1, r1): return P(e1, r1)",  # 1p
    # 2. entity multi-hop
    "P2": "def P2(e1, r1, r2): return P(P(e1, r1), r2)",  # 2p
    "P3": "def P3(e1, r1, r2, r3): return P(P(P(e1, r1), r2), r3)",  # 3p
    # 3. entity and
    "e2i": "def e2i(e1, r1, e2, r2): return And(P(e1, r1), P(e2, r2))",  # 2i
    "e3i": "def e3i(e1, r1, e2, r2, e3, r3): return And3(P(e1, r1), P(e2, r2), P(e3, r3))",  # 3i
    "e2i_P": "def e2i_P(e1, r1, r2, e2, r3): return And(P(P(e1, r1), r2), P(e2, r3))",  # pi
    "P_e2i": "def P_e2i(e1, r1, e2, r2, r3): return P(e2i(e1, r1, e2, r2), r3)",  # ip
    # 5. entity not
    "e2i_NP": "def e2i_NP(e1, r1, r2, e2, r3): return And(Not(P(P(e1, r1), r2)), P(e2, r3))",  # pni
    "e2i_PN": "def e2i_PN(e1, r1, r2, e2, r3): return And(P(P(e1, r1), r2), Not(P(e2, r3)))",  # pin
    "P_e2i_P_NP": "def P_e2i_P_NP(e1, r1, e2, r2, r3): return P(And(P(e1, r1), Not(P(e2, r2))), r3)",  # inp
    "e2i_N": "def e2i_N(e1, r1, e2, r2): return And(P(e1, r1), Not(P(e2, r2)))",  # 2in
    "e3i_N": "def e3i_N(e1, r1, e2, r2, e3, r3): return And3(P(e1, r1), P(e2, r2), Not(P(e3, r3)))",  # 3in
    # 6. entity union
    "e2u": "def e2u(e1, r1, e2, r2): return Or(P(e1, r1), P(e2, r2))",  # 2u
    "P_e2u": "def P_e2u(e1, r1, e2, r2, r3): return P(Or(P(e1, r1), P(e2, r2)), r3)",  # up
    # 7. union-DM
    "e2u_DM": "def e2u_DM(e1, r1, e2, r2): return Not(And(Not(P(e1, r1)), Not(P(e2, r2))))",  # 2u-DM
    "P_e2u_DM": "def P_e2u_DM(e1, r1, e2, r2, r3): return P(Not(And(Not(P(e1, r1)), Not(P(e2, r2)))), r3)",  # up-DM
    # 8. union-DNF
    "e2u_DNF": "def e2u_DNF(e1, r1, e2, r2): return P(e1, r1), P(e2, r2)",  # 2u_DNF
    "P_e2u_DNF": "def P_e2u_DNF(e1, r1, e2, r2, r3): return P(P(e1, r1), r3), P(P(e2, r2), r3)",  # up_DNF
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
