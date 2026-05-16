from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List
from data.filtering.registry import FILTER_REGISTRY, FilterDefinition, FilterResult
from data.filtering.utils import drop_other_columns, require_no_params, require_number_param, require_param

if TYPE_CHECKING:
    from data.plan.engine import EngineContext


# -------------------------
# drop columns whose schema confidence is too low
# -------------------------

def _validate_drop_low_confidence_schema(params: Dict[str, Any]):
    require_number_param(params, "threshold", "drop_low_confidence_schema")


def _filter_drop_low_confidence_schema(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    threshold = require_number_param(params, "threshold", "drop_low_confidence_schema")
    cols = [col for col, entry in ctx.schema.items() if entry.confidence >= threshold]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="schema",
    name="drop_low_confidence_schema",
    definition=FilterDefinition(
        fn=_filter_drop_low_confidence_schema,
        validate_params=_validate_drop_low_confidence_schema,
    ),
)


# -------------------------
# drop columns that were forced by user schema
# -------------------------

def _validate_drop_forced_schema(params: Dict[str, Any]):
    require_no_params(params, "drop_forced_schema")


def _filter_drop_forced_schema(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = [col for col, entry in ctx.schema.items() if not entry.forced]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="schema",
    name="drop_forced_schema",
    definition=FilterDefinition(
        fn=_filter_drop_forced_schema,
        validate_params=_validate_drop_forced_schema,
    ),
)


# -------------------------
# drop columns with multiple candidate schema types
# -------------------------

def _validate_drop_if_multiple_candidates(params: Dict[str, Any]):
    require_no_params(params, "drop_if_multiple_candidates")


def _filter_drop_if_multiple_candidates(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = [col for col, entry in ctx.schema.items() if len(entry.candidates) <= 1]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="schema",
    name="drop_if_multiple_candidates",
    definition=FilterDefinition(
        fn=_filter_drop_if_multiple_candidates,
        validate_params=_validate_drop_if_multiple_candidates,
    ),
)
