from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class TransformStep:
    """
    Single transformation step (column, multi-column, or derived).

    id:       unique identifier for this step (used for DAG-based ordering).
    label:    optional UI label for this step.
    category: registry category (e.g. "missing", "scaling", "encoding", "datetime", "derived", ...)
    name:     transformation name within that category.
    inputs:   list of input column names this step depends on.
    params:   parameter dict passed to the transformation.
    after:    optional list of step IDs that must run before this step
              (used for DAG-based ordering).
    """
    id: str
    label: Optional[str]
    category: str
    name: str
    inputs: List[str]
    params: Dict[str, Any]
    after: Optional[List[str]] = None


@dataclass(frozen=True)
class TransformationPlan:
    """
    Full transformation plan.

    steps: global list of TransformStep, ordered via DAG (toposort) using `after`.
           Each step can be:
             - single-column (inputs=["col"])
             - multi-column (inputs=["col1", "col2", ...])
             - derived (e.g. category="derived", arbitrary inputs)
    """
    steps: List[TransformStep]
