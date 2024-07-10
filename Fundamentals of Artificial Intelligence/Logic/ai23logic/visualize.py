from collections import defaultdict
from field_var import field_var, get_name

def draw(layout):
    return " | ".join(layout)


def draw_solution(layout, solution):
    if layout is None or solution is None:
        return "You didn't find a logic-solution."

    elements = {
        field_var(fruit, i)
        for i, fruit in enumerate(layout, start=1)
        if fruit != "*"
    }
    elements.update(solution.split("&"))

    position_assignment = defaultdict(list)
    # mark all positions in the garden explicitly as empty, so they are picked up as wrongly empty later
    for pos in range(1, len(layout) + 1):
        position_assignment[pos] = []
    for fruit in elements:
        fruit_name = get_name(fruit[:-1])
        fruit_pos = int(fruit[-1])
        position_assignment[fruit_pos].append(fruit_name)

    for pos, fruits in position_assignment.items():
        if pos not in range(1, len(layout) + 1):
            print(f"[WARNING] You assigned fruit(s) to a position outside of the garden: All of {', '.join(fruits)} are assigned to position {pos}.")
        if len(fruits) > 1:
            print(f"[WARNING] You assigned multiple fruits to the same position: All of {', '.join(fruits)} are assigned to position {pos}.")
        if len(fruits) < 1:
            print(f"[WARNING] You didn't assign a fruit to position {pos}.")

    solution_list = [next(iter(fruits), "*") for pos, fruits in position_assignment.items()]
    print(" | ".join(solution_list))
