from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List
from data.filtering.registry import FILTER_REGISTRY, FilterDefinition, FilterResult
from data.filtering.utils import require_list_param, require_no_params, require_param

if TYPE_CHECKING:
    from data.plan.engine import EngineContext


# value‑based row filtering

# -------------------------
# Row filter: keep rows where column > threshold
# -------------------------

def _validate_threshold(params: Dict[str, Any]):
    require_param(params, "threshold", "greater_than")


def _filter_threshold(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = inputs[0]
    thr = require_param(params, "threshold", "greater_than")
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
# Row filter: drop rows where column is null
# -------------------------



def _validate_drop_nulls(params: Dict[str, Any]):
    require_no_params(params, "drop_nulls")


def _filter_drop_nulls(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = inputs[0]
    mask = ctx.df[col].notna()
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="row",
    name="drop_nulls",
    definition=FilterDefinition(
        fn=_filter_drop_nulls,
        validate_params=_validate_drop_nulls,
    ),
)


# -------------------------
# Row filter: drop rows where any input column is null
# -------------------------

def _validate_drop_rows_if_any_null(params: Dict[str, Any]):
    require_no_params(params, "drop_rows_if_any_null")


def _filter_drop_rows_if_any_null(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    if not inputs:
        raise RuntimeError("drop_rows_if_any_null requires at least one input column.")
    for col in inputs:
        if col not in ctx.df.columns:
            raise RuntimeError(f"drop_rows_if_any_null requested unknown column '{col}'.")
    mask = ctx.df[inputs].notna().all(axis=1)
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="row",
    name="drop_rows_if_any_null",
    definition=FilterDefinition(
        fn=_filter_drop_rows_if_any_null,
        validate_params=_validate_drop_rows_if_any_null,
    ),
)


# -------------------------
# Row filter: drop rows where all input columns are null
# -------------------------

def _validate_drop_rows_if_all_null(params: Dict[str, Any]):
    require_no_params(params, "drop_rows_if_all_null")


def _filter_drop_rows_if_all_null(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    if not inputs:
        raise RuntimeError("drop_rows_if_all_null requires at least one input column.")
    for col in inputs:
        if col not in ctx.df.columns:
            raise RuntimeError(f"drop_rows_if_all_null requested unknown column '{col}'.")
    mask = ~ctx.df[inputs].isna().all(axis=1)
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="row",
    name="drop_rows_if_all_null",
    definition=FilterDefinition(
        fn=_filter_drop_rows_if_all_null,
        validate_params=_validate_drop_rows_if_all_null,
    ),
)


# -------------------------
# Row filter: keep rows where column value is in a list of values
# -------------------------

def _validate_in_list(params: Dict[str, Any]):
    require_list_param(params, "values", "in_list")


def _filter_in_list(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = inputs[0]
    values = require_list_param(params, "values", "in_list")
    mask = ctx.df[col].isin(values)
    return FilterResult(drop_rows=mask.tolist())

FILTER_REGISTRY.register(
    category="row",
    name="in_list",
    definition=FilterDefinition(
        fn=_filter_in_list,
        validate_params=_validate_in_list,
    ),
)

# -------------------------
# Row filter: keep rows where column value matches a regex pattern
# -------------------------
import re

def _validate_regex_match(params: Dict[str, Any]):
    pattern = require_param(params, "pattern", "regex_match")
    try:
        re.compile(pattern)
    except re.error:
        raise ValueError("regex_match filter requires a valid regex pattern.")


def _filter_regex_match(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = inputs[0]
    pattern = require_param(params, "pattern", "regex_match")
    mask = ctx.df[col].str.contains(pattern, na=False)
    return FilterResult(drop_rows=mask.tolist())

FILTER_REGISTRY.register(
    category="row",
    name="regex_match",
    definition=FilterDefinition(
        fn=_filter_regex_match,
        validate_params=_validate_regex_match,
    ),
)


#---------------------------
# Row filter: compare to constants
#---------------------------

def _validate_comparison(params: Dict[str, Any]):
    require_param(params, "operator", "comparison")
    require_param(params, "value", "comparison")
    if params["operator"] not in [">", "<", ">=", "<=", "==", "!="]:
        raise ValueError("comparison filter operator must be one of >, <, >=, <=, ==, !=")


def _filter_comparison(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = inputs[0]
    op = require_param(params, "operator", "comparison")
    val = require_param(params, "value", "comparison")
    mask = ctx.df[col].notna()
    if op == ">":
        mask = ctx.df[col] > val
    elif op == "<":
        mask = ctx.df[col] < val
    elif op == ">=":
        mask = ctx.df[col] >= val
    elif op == "<=":
        mask = ctx.df[col] <= val
    elif op == "==":
        mask = ctx.df[col] == val
    elif op == "!=":
        mask = ctx.df[col] != val
    return FilterResult(drop_rows=mask.tolist())

FILTER_REGISTRY.register(
    category="row",
    name="comparison",
    definition=FilterDefinition(
        fn=_filter_comparison,
        validate_params=_validate_comparison,
    ),
)

