from typing import List, Dict, Set
from .plan import ColumnTransformStep


def toposort_steps(steps: List[ColumnTransformStep]) -> List[ColumnTransformStep]:
    """
    Topologically sort steps based on their 'after' dependencies (by category).

    Each step can declare `after=["missing", "outliers"]`, meaning:
    all steps whose category is in that list must run before this one.

    If there is a cycle, a ValueError is raised.
    """
    # Group steps by category for dependency resolution
    by_category: Dict[str, List[ColumnTransformStep]] = {}
    for s in steps:
        by_category.setdefault(s.category, []).append(s)

    # Build graph: node = index in steps list
    n = len(steps)
    deps: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i, step in enumerate(steps):
        after = step.after or []
        for cat in after:
            for j, other in enumerate(steps):
                if other.category == cat:
                    deps[i].add(j)

    # Kahn's algorithm
    result: List[ColumnTransformStep] = []
    no_incoming = [i for i in range(n) if not deps[i]]

    while no_incoming:
        i = no_incoming.pop()
        result.append(steps[i])

        # Remove edges
        for j in range(n):
            if i in deps[j]:
                deps[j].remove(i)
                if not deps[j]:
                    no_incoming.append(j)

    if any(deps[i] for i in range(n)):
        raise ValueError("Cycle detected in column transformation dependencies")

    return result
