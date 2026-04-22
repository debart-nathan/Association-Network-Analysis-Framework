from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class ColumnTransformStep:
    """
    Single transformation step for a column.

    category: registry category (e.g. "missing", "scaling", "encoding", "datetime", ...)
    name:     transformation name within that category
    params:   parameter dict passed to the transformation
    after:    optional list of categories that must run before this step
              (used for DAG-based ordering)
    """
    label: Optional[str]
    category: str
    name: str
    params: Dict[str, Any]
    after: Optional[List[str]] = None


@dataclass(frozen=True)
class ColumnTransformSpec:
    """
    Declarative specification of all transformations to apply to a single column.

    steps: ordered or dependency-resolved list of ColumnTransformStep.
    """
    steps: List[ColumnTransformStep]


@dataclass(frozen=True)
class DerivedSpec:
    """
    Declarative specification of a derived (multi-column) transformation.

    new_col: name of the resulting column (if the transform returns a Series).
             If the transform returns a DataFrame, this can be ignored or used
             as a prefix by the transform itself.
    name:    transformation name in the "derived" category.
    params:  parameter dict passed to the transformation.
    """
    new_col: str
    name: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class TransformationPlan:
    """
    Full transformation plan.

    columns: mapping from column name to its column-level spec.
    derived: list of multi-column / expression-based transformations.
    """
    columns: Dict[str, ColumnTransformSpec]
    derived: Optional[List[DerivedSpec]] = None
