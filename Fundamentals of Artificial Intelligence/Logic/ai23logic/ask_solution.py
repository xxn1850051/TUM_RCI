def ask_solution(kb):
    sat_assignment = kb.ask_sat()
    # return only positive assignments
    return "&".join(str(sym) for sym in sat_assignment if sym.op != "~")
