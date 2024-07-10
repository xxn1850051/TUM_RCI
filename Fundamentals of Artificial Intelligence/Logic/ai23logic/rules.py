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
            if i == 1:
                kb.append(f'{field_var(self.fruit_a, i)} ==> {field_var(self.fruit_b, i+1)}')
            elif i == layout_size:
                kb.append(f'{field_var(self.fruit_a, i)} ==> {field_var(self.fruit_b, i-1)}')
            else:
                kb.append(f'{field_var(self.fruit_a, i)} ==> {field_var(self.fruit_b, i-1)} | {field_var(self.fruit_b, i+1)}')
                
        return kb
        
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
            if self.init_layout[i-1] != '*':
                kb.append(f'{field_var(self.init_layout[i-1], i)}')
        
        return kb

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
                rest_fruit = [rf for rf in self.fruit_types if rf != f]
                kb.append(f"{field_var(f, i)} ==> ~({ '|'.join(field_var(nf, i) for nf in rest_fruit) })")       
        
        return kb

        pass

        # TODO End
    
"""
class OnePlantPerCellConstraint(Constraint):
    #Each cell can only hold one plant.

    fruit_types: List[str]

    def formalize(self, layout_size: int) -> List[str]:
        assert len(self.fruit_types) == layout_size, "There must be as many fruits as there are spots in the layout."

        kb = []

        for i in range(1,layout_size+1):
            kb.append(
                "(" + " | ".join(field_var(fruit, i) for fruit in self.fruit_types) + ")"
            )
          
        for i in range(layout_size):
            for j in range(layout_size):
                for k in range(layout_size):
                    if(j!=i):
                
                        kb.append(
                            "(" + field_var(self.fruit_types[i], k+1) + ")"
                            + " ==> " + "~(" + field_var(self.fruit_types[j], k+1) + ")"
                        )

        return kb
"""

@dataclass(frozen=True)
class OneCellPerPlantConstraint(Constraint):
    """Each plant can only be planted in one cell."""

    fruit_types: List[str]

    def formalize(self, layout_size: int) -> List[str]:
        assert len(self.fruit_types) == layout_size, "There must be as many fruits as there are spots in the layout."

        # TODO Start
        kb = []
        location = range(1, layout_size+1)
        
#         for i in location:
#             for f in self.fruit_types:
#                 for j in range(1,layout_size+1):
#                     if j!= i:
#                         kb.append(f'{field_var(f, j)}==>~{field_var(f, i)}')   
        
        for current_location in location:
            for current_fruit in self.fruit_types:
                other_fruits = list(filter(lambda f: f != current_fruit, self.fruit_types))
                other_locations = list(range(1, layout_size + 1))
                other_locations.remove(current_location)
                
                kb.append(
                    f"{field_var(current_fruit, current_location)}==>"
                    + "&".join(
                        f"({ '|'.join(field_var(other_fruit, other_location) for other_location in other_locations) })"
                        for other_fruit in other_fruits
                    )
                )
                                        
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
            f'{field_var(self.fruit_a, self.position)} | {field_var(self.fruit_b, self.position)}')
                
        return kb

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
            f'{field_var(self.fruit_a, self.position_a)} ==> {field_var(self.fruit_b, self.position_b)}')
        
        return kb

        # TODO End
