from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from data.plan.base_step import BaseStep


@dataclass(frozen=True)
class SchemaCastStep(BaseStep):
    """
    A DAG step representing a schema type cast.

    Inherits:
      - id
      - label
      - step_type = "schema_cast"
      - category
      - name
      - inputs
      - params
      - after

    Expected behavior:
      - no new columns
      - no dropped columns
      - no row filtering
      - only schema + metadata updates
    """

    @staticmethod
    def create(
        id: str,
        base: str,
        subtype: str,
        column: str,
        label: Optional[str] = None,
        after: Optional[List[str]] = None,
    ) -> "SchemaCastStep":
        """
        Factory method to build a SchemaCastStep.

        Parameters:
            id: unique step id
            base: target base type (e.g. "numeric", "categorical")
            subtype: target subtype (e.g. "continuous", "ordinal")
            column: column to cast
        """
        return SchemaCastStep(
            id=id,
            label=label,
            step_type="schema_cast",
            category="schema",
            name="cast",
            inputs=[column],
            params={
                "base": base,
                "subtype": subtype,
            },
            after=after,
        )
