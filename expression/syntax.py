"""
@date: 2022/3/21
@description: null
"""

def Pe(e1, r1, t1): return 0
def Pt(e1, r1, e2): return 0
def after(t): return 0
def before(t): return 0
def next(t): return 0
def And(q1, q2): return 0
def And3(q1, q2, q3): return 0
def Not(t): return 0
def Or(q1, q2): return 0
def TimeAnd(q1, q2): return 0
def TimeAnd3(q1, q2, q3): return 0
def TimeNot(t): return 0
def TimeOr(q1, q2): return 0


# 1. 1-hop Pe and Pt, manually
# 2. entity multi-hop
def Pe2(e1, r1, t1, r2, t2): return Pe(Pe(e1, r1, t1), r2, t2)  # 2p
def Pe3(e1, r1, t1, r2, t2, r3, t3): return Pe(Pe(Pe(e1, r1, t1), r2, t2), r3, t3)  # 3p
# 3. time multi-hop
def Pt_lPe(e1, r1, t1, r2, e2): return Pt(Pe(e1, r1, t1), r2, e2)  # l for left (as head entity)
def Pt_rPe(e1, r1, e2, r2, t1): return Pt(e1, r1, Pe(e2, r2, t1))  # r for right (as tail entity)
def Pe_Pt(e1, r1, e2, r2, e3): return Pe(e1, r1, Pt(e2, r2, e3))  # at
def Pe_aPt(e1, r1, e2, r2, e3): return Pe(e1, r1, after(Pt(e2, r2, e3)))  # a for after
def Pe_bPt(e1, r1, e2, r2, e3): return Pe(e1, r1, before(Pt(e2, r2, e3)))  # b for before
def Pe_nPt(e1, r1, e2, r2, e3): return Pe(e1, r1, next(Pt(e2, r2, e3)))  # n for next
# 4. entity and & time and
def e2i(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Pe(e2, r2, t2))  # 2i
def e3i(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Pe(e3, r3, t3))  # 3i
def t2i(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), Pt(e3, r2, e4))  # t-2i
def t3i(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), Pt(e5, r3, e6))  # t-3i
# 5. complex time and
def e2i_Pe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Pe(e2, r3, t3))  # pi
def Pe_e2i(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(e2i(e1, r1, t1, e2, r2, t2), r3, t3)  # ip
def Pt_le2i(e1, r1, t1, e2, r2, t2, r3, e3): return Pt(e2i(e1, r1, t1, e2, r2, t2), r3, e3)  # mix ip
def Pt_re2i(e1, r1, e2, r2, t1, e3, r3, t2): return Pt(e1, r1, e2i(e2, r2, t1, e3, r3, t2))  # mix ip
def t2i_Pe(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), Pt(e3, r3, e4))  # t-pi
def Pe_t2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, t2i(e2, r2, e3, e4, r3, e5))  # t-ip
def Pe_at2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, after(t2i(e2, r2, e3, e4, r3, e5)))
def Pe_bt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, before(t2i(e2, r2, e3, e4, r3, e5)))
def Pe_nt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, next(t2i(e2, r2, e3, e4, r3, e5)))
def between(e1, r1, e2, e3, r2, e4): return TimeAnd(after(Pt(e1, r1, e2)), before(Pt(e3, r2, e4)))  # between(t1, t2) == after t1 and before t2
# 5. entity not
def e2i_NPe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Not(Pe(Pe(e1, r1, t1), r2, t2)), Pe(e2, r3, t3))  # pni = e2i_N(Pe(e1, r1, t1), r2, t2, e2, r3, t3)
def e2i_PeN(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Not(Pe(e2, r3, t3)))  # pin
def Pe_e2i_Pe_NPe(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2))), r3, t3)  # inp
def e2i_N(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2)))  # 2in
def e3i_N(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Not(Pe(e3, r3, t3)))  # 3in
# 6. time not
def t2i_NPt(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(TimeNot(Pt(Pe(e1, r1, t1), r2, e2)), Pt(e3, r3, e4))  # t-pni
def t2i_PtN(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), TimeNot(Pt(e3, r3, e4)))  # t-pin
def Pe_t2i_PtPe_NPt(e1, r1, e2, r2, t2, r3, e3, e4, r4, e5): return Pe(e1, r1, TimeAnd(Pt(Pe(e2, r2, t2), r3, e3), TimeNot(Pt(e4, r4, e5))))  # t-inp
def t2i_N(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), TimeNot(Pt(e3, r2, e4)))  # t-2in
def t3i_N(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), TimeNot(Pt(e5, r3, e6)))  # t-3in
# 7. entity union & time union
def e2u(e1, r1, t1, e2, r2, t2): return Or(Pe(e1, r1, t1), Pe(e2, r2, t2))  # 2u
def Pe_e2u(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Or(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)  # up
def t2u(e1, r1, e2, e3, r2, e4): return TimeOr(Pt(e1, r1, e2), Pt(e3, r2, e4))  # t-2u
def Pe_t2u(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, TimeOr(Pt(e2, r2, e3), Pt(e4, r3, e5)))  # t-up
# 8. union-DM
def e2u_DM(e1, r1, t1, e2, r2, t2): return Not(And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2))))  # 2u-DM
def Pe_e2u_DM(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Not(And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2)))), r3, t3)  # up-DM
def t2u_DM(e1, r1, e2, e3, r2, e4): return TimeNot(TimeAnd(TimeNot(Pt(e1, r1, e2)), TimeNot(Pt(e3, r2, e4))))  # t-2u-DM
def Pe_t2u_DM(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, TimeNot(TimeAnd(TimeNot(Pt(e2, r2, e3)), TimeNot(Pt(e4, r3, e5)))))  # t-up-DM
# 9. union-DNF
def e2u_DNF(e1, r1, t1, e2, r2, t2): return Pe(e1, r1, t1), Pe(e2, r2, t2)  # 2u_DNF
def Pe_e2u_DNF(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Pe(e1, r1, t1), r3, t3), Pe(Pe(e2, r2, t2), r3, t3)  # up_DNF
def t2u_DNF(e1, r1, e2, e3, r2, e4): return Pt(e1, r1, e2), Pt(e3, r2, e4)  # t-2u_DNF
def Pe_t2u_DNF(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, Pt(e2, r2, e3)), Pe(e1, r1, Pt(e4, r3, e5))  # t-up_DNF
