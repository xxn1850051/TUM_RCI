from logic import conjuncts, to_cnf, disjuncts

from dpll import fast_dpll, fast_dpll_sat


class DpllPropKB:
    def __init__(self):
        self.clauses = set()

    def tell(self, sentence):
        self.clauses.update(self._to_clauses(sentence))

    def has_contradicting_knowledge(self):
        return not fast_dpll(frozenset(self.clauses))

    def ask(self, prop):
        to_prove = self._to_clauses(~prop).union(self.clauses)

        if fast_dpll(to_prove):
            return False
        else:
            self.tell(prop)
            return True

    def ask_sat(self, prop=None):
        to_prove = self._to_clauses(~prop).union(self.clauses) if prop else frozenset(self.clauses)

        sat = fast_dpll_sat(to_prove, needed_assignments=1)
        return next(iter(sat), None)

    @staticmethod
    def _to_clauses(sentence):
        return frozenset(
            frozenset(disjuncts(clause))
            for clause in conjuncts(to_cnf(sentence))
        )

