from typing import List, Dict, Set
from dataclasses import dataclass

from data.plan.base_step import BaseStep


@dataclass(frozen=True)
class DataPlan:
    """
    Full data plan.

    steps: global list of steps, ordered via DAG (toposort) using `after`.
           Steps can be:
             - SchemaCastStep
             - TransformStep
             - FilterStep
    """
    steps: List[BaseStep]


def toposort_steps(steps: List[BaseStep]) -> List[BaseStep]:
    """
    Topologically sort BaseStep objects based on their 'after' dependencies.

    Each step can declare:
        after=["step_a", "step_b"]

    meaning:
        step_a and step_b must run BEFORE this step.

    If there is a cycle, a ValueError is raised.
    """

    # 1. Build index map
    id_to_index: Dict[str, int] = {}
    for i, step in enumerate(steps):
        if step.id in id_to_index:
            raise ValueError(f"Duplicate step id detected: '{step.id}'")
        id_to_index[step.id] = i

    n = len(steps)

    # 2. Build dependency graph
    deps: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i, step in enumerate(steps):
        after_ids = step.after or []
        for dep_id in after_ids:
            if dep_id not in id_to_index:
                raise ValueError(
                    f"Step '{step.id}' depends on unknown step id '{dep_id}'"
                )
            deps[i].add(id_to_index[dep_id])

    # 3. Kahn's algorithm for topological sorting
    result: List[BaseStep] = []
    no_incoming = [i for i in range(n) if not deps[i]]

    while no_incoming:
        i = no_incoming.pop()
        result.append(steps[i])

        # Remove edges from i → others
        for j in range(n):
            if i in deps[j]:
                deps[j].remove(i)
                if not deps[j]:
                    no_incoming.append(j)

    # 4. Detect cycles
    if any(deps[i] for i in range(n)):
        raise ValueError("Cycle detected in plan dependencies")

    return result
