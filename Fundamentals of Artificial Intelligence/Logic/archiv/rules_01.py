from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from field_var import field_var


class Constraint(ABC):
    @abstractmethod
    def formalize(self, layout_size: int) -> List[str]:
        """Formalize the constraint as logic sentences.

        :param layout_size: The size of the layout.
        :return: A list of strings, each representing a logic sentence.
        """
        pass


@dataclass(frozen=True)
class NextToConstraint(Constraint):
    """A must be next to B."""

    fruit_a: str
    fruit_b: str

    def formalize(self, layout_size: int) -> List[str]:

        # TODO Start
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            kb.append(
                f'{field_var(self.fruit_b, i)} ==> {field_var(self.fruit_a, i-1)} | {field_var(self.fruit_a, i+1)}')
            kb.append(
                f'{field_var(self.fruit_a, i)} ==> {field_var(self.fruit_b, i-1)} | {field_var(self.fruit_b, i+1)}')
        
        return kb
       
        pass

        # TODO End


@dataclass(frozen=True)
class InitialLayoutConstraint(Constraint):
    """The initial layout has to be respected."""

    init_layout: List[str]

    def formalize(self, layout_size: int) -> List[str]:
        assert len(
            self.init_layout) == layout_size, "The initial layout has to be of the same size as the layout we are formalizing for."

        # TODO Start
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            kb.append(
                f'{init_layout[i-1] != '*'} ==> {field_var(init_layout[i-1], i)}')
        
        return kb     
        
        pass

        # TODO End


@dataclass(frozen=True)
class OnePlantPerCellConstraint(Constraint):
    """Each cell can only hold one plant."""

    fruit_types: List[str]

    def formalize(self, layout_size: int) -> List[str]:
        assert len(self.fruit_types) == layout_size, "There must be as many fruits as there are spots in the layout."

        # TODO Start
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            for f in self.fruit_types:
                if field_var(f, i):
                    for j in range(i, layout_size + 1):
                        f'{field_var(f, i)}==>~{field_var(f, j)}'                 
        
        return kb
        
        pass

        # TODO End


@dataclass(frozen=True)
class OneCellPerPlantConstraint(Constraint):
    """Each plant can only be planted in one cell."""

    fruit_types: List[str]

    def formalize(self, layout_size: int) -> List[str]:
        assert len(self.fruit_types) == layout_size, "There must be as many fruits as there are spots in the layout."

        # TODO Start
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            for f in self.fruit_types:
                if field_var(f, i):
                    for j in range(i, layout_size + 1):
                        f'{field_var(f, i)}==>~{field_var(f, j)}'                 
        
        return kb

        pass

        # TODO End


@dataclass(frozen=True)
class NotNextToConstraint(Constraint):
    """A cannot be next to B."""

    fruit_a: str
    fruit_b: str

    def formalize(self, layout_size: int) -> List[str]:

        # TODO Start
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            kb.append(
                f'{field_var(self.fruit_b, i)} ==> ~{field_var(self.fruit_a, i-1)} & ~{field_var(self.fruit_a, i+1)}')
            kb.append(
                f'{field_var(self.fruit_a, i)} ==> ~{field_var(self.fruit_b, i-1)} & ~{field_var(self.fruit_b, i+1)}')
        
        return kb

        pass

        # TODO End


@dataclass(frozen=True)
class ToTheRightConstraint(Constraint):
    """A must be next to and right of B."""

    fruit: str
    must_be_right_of: str

    def formalize(self, layout_size: int) -> List[str]:

        # TODO Start
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            kb.append(
                f'{field_var(self.must_be_right_of, i)} ==> {field_var(self.fruit, i+1)}')
            kb.append(
                f'{field_var(self.fruit, i)} ==> {field_var(self.must_be_right_of, i-1)}')
        
        return kb

        pass

        # TODO End


@dataclass(frozen=True)
class ToTheLeftConstraint(Constraint):
    """A must be next to and left of B."""

    fruit: str
    must_be_left_of: str

    def formalize(self, layout_size: int) -> List[str]:

        # TODO Start
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            kb.append(
                f'{field_var(self.must_be_left_of, i)} ==> {field_var(self.fruit, i-1)}')
            kb.append(
                f'{field_var(self.fruit, i)} ==> {field_var(self.must_be_left_of, i+1)}')
        
        return kb

        pass

        # TODO End


@dataclass(frozen=True)
class IfThenNotNextToConstraint(Constraint):
    """If A is left to B, then A cannot be next to C."""

    fruit_a: str
    fruit_b: str
    cannot_be_next_to: str  # fruit C

    def formalize(self, layout_size: int) -> List[str]:
        kb = []
        location = range(1, layout_size + 1)
        for i in location:
            kb.append(
                "(" + field_var(self.fruit_a, i) + "&" + field_var(self.fruit_b, i + 1) + ")"
                + "==>" + "~(" +
                field_var(self.fruit_a, i) + "&" + "(" + field_var(self.cannot_be_next_to, i + 1) + "|" + field_var(
                    self.cannot_be_next_to, i - 1) + ")" + ")")
            kb.append(
                "(" + field_var(self.fruit_a, i) + "&" + field_var(self.fruit_b, i + 1) + ")"
                + "==>" + "~(" +
                field_var(self.cannot_be_next_to, i) + "&" + "(" + field_var(self.fruit_a, i + 1) + "|" + field_var(
                    self.fruit_a, i - 1) + ")" + ")")

        return kb


@dataclass(frozen=True)
class EitherOrConstraint(Constraint):
    """A or B must be in position x."""

    fruit_a: str
    fruit_b: str
    position: int

    def formalize(self, layout_size: int) -> List[str]:

        # TODO Start
        kb = []
        kb.append(
            f'{field_var(self.fruit_a, position)} | {field_var(self.fruit_b, position)}')
        
        return kb

        pass

        # TODO End


@dataclass(frozen=True)
class IfThenConstraint(Constraint):
    """If A is in position x, then B must be in position y."""

    fruit_a: str
    position_a: int  # x
    fruit_b: str
    position_b: int  # y

    def formalize(self, layout_size: int) -> List[str]:

        # TODO Start
        kb = []
        kb.append(
            f'{field_var(self.fruit_a, position_a)} ==> {field_var(self.fruit_b, position_b)}')
        
        return kb

        pass

        # TODO End
