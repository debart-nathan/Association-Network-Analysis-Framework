from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, List, Dict, Any

if TYPE_CHECKING:
    from data.plan.engine import EngineContext


#-----------------------------------------------------------
# FilterResult
#-----------------------------------------------------------

@dataclass(frozen=True)
class FilterResult:
    """
    Result of a filtering step.

    drop_rows: boolean mask (same length as df) or None
    drop_columns: list of column names to drop
    new_schema: optional schema updates (rare for filters)
    new_metadata: optional metadata updates
    """

    drop_rows: Optional[List[bool]] = None
    drop_columns: List[str] = field(default_factory=list)
    new_schema: Dict[str, Any] = field(default_factory=dict)
    new_metadata: Dict[str, dict] = field(default_factory=dict)

    @staticmethod
    def empty() -> "FilterResult":
        return FilterResult()


# ============================================================
# FilterDefinition
# ============================================================

@dataclass(frozen=True)
class FilterDefinition:
    """
    Defines a filter operation.

    Attributes:
        fn: function(ctx, inputs, params) -> FilterResult
        validate_params: optional parameter validator
    """
    fn: Callable[[EngineContext, List[str], Dict[str, Any]], FilterResult]
    validate_params: Optional[Callable[[Dict[str, Any]], None]] = None


# ============================================================
# FilterRegistry
# ============================================================

class FilterRegistry:
    """
    Registry for all filtering operations.

    Structure:
        registry[category][name] = FilterDefinition
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, FilterDefinition]] = {}

    def register(self, category: str, name: str, definition: FilterDefinition):
        if category not in self._registry:
            self._registry[category] = {}

        if name in self._registry[category]:
            raise ValueError(
                f"Filter '{category}:{name}' is already registered."
            )

        self._registry[category][name] = definition

    def get(self, category: str, name: str) -> FilterDefinition:
        try:
            return self._registry[category][name]
        except KeyError:
            raise ValueError(
                f"Unknown filter '{category}:{name}'. "
                f"Available: {list(self._registry.get(category, {}).keys())}"
            )

    def all(self):
        return self._registry


# ============================================================
# Global registry instance
# ============================================================

FILTER_REGISTRY = FilterRegistry()

# Import filtering category modules to populate the registry.
# This keeps filter registration self-contained when the registry is imported.
import data.filtering.categories  # noqa: F401



