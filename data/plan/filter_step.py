from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from data.plan.base_step import BaseStep


@dataclass(frozen=True)
class FilterStep(BaseStep):
    """
    A DAG step representing a filtering operation.

    Inherits:
      - id
      - label
      - step_type = "filter"
      - category
      - name
      - inputs
      - params
      - after
    """

    @staticmethod
    def create(
        id: str,
        category: str,
        name: str,
        inputs: List[str],
        params: Dict[str, Any],
        label: Optional[str] = None,
        after: Optional[List[str]] = None,
    ) -> "FilterStep":
        """
        Factory method to build a FilterStep.

        Parameters:
            id: unique step id
            category: filter category (e.g. "row", "column", "frequency")
            name: filter name within the category
            inputs: columns used by the filter
            params: filter parameters
        """
        return FilterStep(
            id=id,
            label=label,
            step_type="filter",
            category=category,
            name=name,
            inputs=inputs,
            params=params,
            after=after,
        )
