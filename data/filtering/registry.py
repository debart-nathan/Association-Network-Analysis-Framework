from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any

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


# ============================================================
# Example built‑in filters (optional)
# ============================================================

# -------------------------
# Row filter: drop rows where column is null
# -------------------------

def _validate_drop_nulls(params: Dict[str, Any]):
    if params:
        raise ValueError("drop_nulls filter takes no parameters.")


def _filter_drop_nulls(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = inputs[0]
    mask = ctx.df[col].notna()
    return FilterResult(
        drop_rows=mask.tolist(),
        drop_columns=[],
        new_schema={},
        new_metadata={},
    )


FILTER_REGISTRY.register(
    category="row",
    name="drop_nulls",
    definition=FilterDefinition(
        fn=_filter_drop_nulls,
        validate_params=_validate_drop_nulls,
    ),
)


# -------------------------
# Row filter: keep rows where column > threshold
# -------------------------

def _validate_threshold(params: Dict[str, Any]):
    if "threshold" not in params:
        raise ValueError("threshold filter requires param 'threshold'.")


def _filter_threshold(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = inputs[0]
    thr = params["threshold"]
    mask = ctx.df[col] > thr
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="row",
    name="greater_than",
    definition=FilterDefinition(
        fn=_filter_threshold,
        validate_params=_validate_threshold,
    ),
)


# -------------------------
# Column filter: drop columns by prefix
# -------------------------

def _validate_prefix(params: Dict[str, Any]):
    if "prefix" not in params:
        raise ValueError("drop_by_prefix requires param 'prefix'.")


def _filter_drop_by_prefix(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    prefix = params["prefix"]
    cols = [c for c in ctx.df.columns if c.startswith(prefix)]
    return FilterResult(drop_columns=cols)


FILTER_REGISTRY.register(
    category="column",
    name="drop_by_prefix",
    definition=FilterDefinition(
        fn=_filter_drop_by_prefix,
        validate_params=_validate_prefix,
    ),
)
