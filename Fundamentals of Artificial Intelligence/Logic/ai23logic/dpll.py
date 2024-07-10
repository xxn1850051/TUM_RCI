def neg_sym(sym):
    if sym.op == '~':
        return sym.args[0]
    else:
        return ~sym


def cnf_filter_assumption(clauses, pos_asspts, neg_asspts):
    return frozenset(
        clause.difference(neg_asspts)
        for clause in clauses
        if not clause.intersection(pos_asspts)
    )


def fast_dpll(clauses):
    return len(_fast_dpll(clauses, frozenset())) > 0


def fast_dpll_sat(clauses, needed_assignments=1):
    return _fast_dpll(clauses, frozenset(), needed_assignments=needed_assignments)


def _fast_dpll(clauses, sat_assignment, needed_assignments=1):
    if needed_assignments < 0:
        # stop, as we don't need further assignments
        return frozenset()

    if frozenset() in clauses:
        return frozenset()  # not satisfiable
    elif len(clauses) == 0:
        # using tuple to make sure that this is a singleton set, even if sat_assignment is iterable
        return frozenset((sat_assignment,))  # satisfiable using sat_assignment

    # check if any symbols are mandatory, because they are the only literal left to satisfy some clause
    mandatory_syms = frozenset(
        next(iter(clause))  # get the first (and only) literal in the clause
        for clause in clauses
        if len(clause) == 1
    )
    if len(mandatory_syms) > 0:
        forbidden_syms = frozenset(neg_sym(sym) for sym in mandatory_syms)
        if forbidden_syms.intersection(mandatory_syms):
            # we found a contradiction
            return frozenset()  # not satisfiable

        return _fast_dpll(
            cnf_filter_assumption(clauses, mandatory_syms, forbidden_syms),
            sat_assignment.union(mandatory_syms),
            needed_assignments=needed_assignments,
        )

    # otherwise, select the most common symbol
    sym_count = {}
    for clause in clauses:
        for next_sym in clause:
            sym_count[next_sym] = sym_count.get(next_sym, 0) + 1

    next_sym = max(sym_count.keys(), key=sym_count.get)
    negated_next_sym = neg_sym(next_sym)
    assume_pos = _fast_dpll(
        cnf_filter_assumption(clauses, {next_sym}, {negated_next_sym}),
        sat_assignment.union({next_sym}),
        needed_assignments=needed_assignments,
    )
    assume_neg = _fast_dpll(
        cnf_filter_assumption(clauses, {negated_next_sym}, {next_sym}),
        sat_assignment.union({negated_next_sym}),
        needed_assignments=needed_assignments - len(assume_pos),
    )
    return assume_pos.union(assume_neg)
