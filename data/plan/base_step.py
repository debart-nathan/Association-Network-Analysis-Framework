from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class BaseStep:
    """
    Base class for all DAG steps (schema cast, transform, filter).

    Every step has:
      - id: unique identifier
      - label: optional UI label
      - step_type: "schema_cast", "transform", "filter"
      - inputs: columns this step depends on
      - params: configuration for the step
      - after: explicit step dependencies
    """
    id: str
    label: Optional[str]
    step_type: str
    category: str
    name: str
    inputs: List[str]
    params: Dict[str, Any]
    after: Optional[List[str]] = None

    def __repr__(self):
        return (
            f"{self.step_type.capitalize()}Step("
            f"id={self.id!r}, category={self.category!r}, name={self.name!r}, inputs={self.inputs}, after={self.after})"
        )
