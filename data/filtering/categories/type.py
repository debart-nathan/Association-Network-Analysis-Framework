from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List
from data.filtering.registry import FILTER_REGISTRY, FilterDefinition, FilterResult
from data.filtering.utils import drop_other_columns, require_list_param, require_param

if TYPE_CHECKING:
    from data.plan.engine import EngineContext


# -------------------------
# keep only columns with a given base type
# -------------------------

def _validate_keep_base_type(params: Dict[str, Any]):
    if "base_type" not in params:
        raise ValueError("keep_base_type filter requires param 'base_type'.")


def _filter_keep_base_type(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    base_type = require_param(params, "base_type", "keep_base_type")
    cols = [col for col, entry in ctx.schema.items() if entry.base == base_type]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="type",
    name="keep_base_type",
    definition=FilterDefinition(
        fn=_filter_keep_base_type,
        validate_params=_validate_keep_base_type,
    ),
)


# -------------------------
# drop columns with a given base type
# -------------------------

def _validate_drop_base_type(params: Dict[str, Any]):
    if "base_types" not in params:
        raise ValueError("drop_base_type filter requires param 'base_types'.")
    if not isinstance(params["base_types"], list):
        raise ValueError("drop_base_type 'base_types' must be a list.")


def _filter_drop_base_type(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    base_types = set(require_list_param(params, "base_types", "drop_base_type"))
    cols = [col for col, entry in ctx.schema.items() if entry.base not in base_types]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="type",
    name="drop_base_type",
    definition=FilterDefinition(
        fn=_filter_drop_base_type,
        validate_params=_validate_drop_base_type,
    ),
)


# -------------------------
# keep only columns with a given subtype
# -------------------------

def _validate_keep_subtype(params: Dict[str, Any]):
    if "subtype" not in params:
        raise ValueError("keep_subtype filter requires param 'subtype'.")


def _filter_keep_subtype(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    subtype = require_param(params, "subtype", "keep_subtype")
    cols = [col for col, entry in ctx.schema.items() if entry.subtype == subtype]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="type",
    name="keep_subtype",
    definition=FilterDefinition(
        fn=_filter_keep_subtype,
        validate_params=_validate_keep_subtype,
    ),
)


# -------------------------
# drop columns with a given subtype
# -------------------------

def _validate_drop_subtype(params: Dict[str, Any]):
    if "subtypes" not in params:
        raise ValueError("drop_subtype filter requires param 'subtypes'.")
    if not isinstance(params["subtypes"], list):
        raise ValueError("drop_subtype 'subtypes' must be a list.")


def _filter_drop_subtype(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    subtypes = set(require_list_param(params, "subtypes", "drop_subtype"))
    cols = [col for col, entry in ctx.schema.items() if entry.subtype not in subtypes]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="type",
    name="drop_subtype",
    definition=FilterDefinition(
        fn=_filter_drop_subtype,
        validate_params=_validate_drop_subtype,
    ),
)


# -------------------------
# drop columns with unknown schema type
# -------------------------

def _filter_drop_unknown(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = [col for col, entry in ctx.schema.items() if entry.base != "unknown"]
    return FilterResult(drop_columns=[col for col in ctx.df.columns if col not in cols])


FILTER_REGISTRY.register(
    category="type",
    name="drop_unknown",
    definition=FilterDefinition(
        fn=_filter_drop_unknown,
    ),
)


# -------------------------
# drop structured columns
# -------------------------

def _filter_drop_structured(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = [col for col, entry in ctx.schema.items() if entry.base != "structured"]
    return FilterResult(drop_columns=[col for col in ctx.df.columns if col not in cols])


FILTER_REGISTRY.register(
    category="type",
    name="drop_structured",
    definition=FilterDefinition(
        fn=_filter_drop_structured,
    ),
)
