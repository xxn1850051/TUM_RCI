from typing import List, Iterator

from rules import Constraint, InitialLayoutConstraint, OnePlantPerCellConstraint, OneCellPerPlantConstraint


def generate_knowledge(init_layout: List[str], rule_list: List[Constraint], to_be_assigned: List[str]) -> Iterator[str]:
    """Generate the knowledge base for the given problem.

    :param init_layout: the initial layout of the garden
    :param rule_list: the list of rules to consider
    :param to_be_assigned: the list of fruits to be assigned
    :return: an iterator over the sentences for the knowledge base
    """
    # add initial constraints to the list of rules
    fruit_types = [fruit for fruit in init_layout if fruit != "*"] + to_be_assigned
    all_rules = rule_list + [
        InitialLayoutConstraint(init_layout),
        OnePlantPerCellConstraint(fruit_types),
        OneCellPerPlantConstraint(fruit_types),
    ]

    # formalize all rules
    location_num = len(init_layout)
    for constraint in all_rules:
        yield from constraint.formalize(location_num)
