from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List
import pandas as pd
from data.filtering.registry import FILTER_REGISTRY, FilterDefinition, FilterResult
from data.filtering.utils import (
    drop_other_columns,
    reject_unknown_params,
    require_int_param,
    require_list_param,
    require_no_params,
    require_number_param,
    require_param,
    validate_columns_exist,
)

if TYPE_CHECKING:
    from data.plan.engine import EngineContext

# -------------------------
# Row filter: drop rows containing invalid values
# -------------------------

def _validate_invalid_values(params: Dict[str, Any]):
    require_list_param(params, "columns", "invalid_values")
    require_list_param(params, "values", "invalid_values")


def _filter_invalid_values(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    columns = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "invalid_values"),
        "invalid_values",
    )
    values = require_list_param(params, "values", "invalid_values")
    mask = pd.Series(True, index=ctx.df.index)
    for col in columns:
        mask &= ~ctx.df[col].isin(values)
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="quality",
    name="invalid_values",
    definition=FilterDefinition(
        fn=_filter_invalid_values,
        validate_params=_validate_invalid_values,
    ),
)


# -------------------------
# Row filter: remove duplicate rows
# -------------------------

def _validate_drop_duplicates(params: Dict[str, Any]):
    reject_unknown_params(params, {"subset", "keep"}, "drop_duplicates")
    if "subset" in params:
        require_list_param(params, "subset", "drop_duplicates")
    if "keep" in params and params["keep"] not in ("first", "last", "none"):
        raise ValueError("drop_duplicates 'keep' must be one of 'first', 'last', or 'none'.")


def _filter_drop_duplicates(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    subset = params.get("subset")
    keep = params.get("keep", "first")

    if keep == "none":
        keep_arg = False
    else:
        keep_arg = keep

    mask = ~ctx.df.duplicated(subset=subset, keep=keep_arg)
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="quality",
    name="drop_duplicates",
    definition=FilterDefinition(
        fn=_filter_drop_duplicates,
        validate_params=_validate_drop_duplicates,
    ),
)


# -------------------------
# Row filter: drop rows with invalid dates
# -------------------------

def _validate_drop_invalid_dates(params: Dict[str, Any]):
    require_param(params, "column", "drop_invalid_dates")


def _filter_drop_invalid_dates(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = params["column"]
    mask = pd.to_datetime(ctx.df[col], errors="coerce").notna()
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="quality",
    name="drop_invalid_dates",
    definition=FilterDefinition(
        fn=_filter_drop_invalid_dates,
        validate_params=_validate_drop_invalid_dates,
    ),
)


# -------------------------
# Row filter: drop rows with invalid numeric values
# -------------------------

def _validate_drop_invalid_numeric(params: Dict[str, Any]):
    require_param(params, "column", "drop_invalid_numeric")


def _filter_drop_invalid_numeric(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    col = params["column"]
    mask = pd.to_numeric(ctx.df[col], errors="coerce").notna()
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="quality",
    name="drop_invalid_numeric",
    definition=FilterDefinition(
        fn=_filter_drop_invalid_numeric,
        validate_params=_validate_drop_invalid_numeric,
    ),
)


# -------------------------
# Row filter: drop rows with too many null values
# -------------------------

def _validate_drop_rows_with_many_nulls(params: Dict[str, Any]):
    require_number_param(params, "threshold", "drop_rows_with_many_nulls")


def _filter_drop_rows_with_many_nulls(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    threshold = params["threshold"]
    if isinstance(threshold, float):
        mask = ctx.df.isna().mean(axis=1) <= threshold
    else:
        mask = ctx.df.isna().sum(axis=1) <= int(threshold)
    return FilterResult(drop_rows=mask.tolist())


FILTER_REGISTRY.register(
    category="quality",
    name="drop_rows_with_many_nulls",
    definition=FilterDefinition(
        fn=_filter_drop_rows_with_many_nulls,
        validate_params=_validate_drop_rows_with_many_nulls,
    ),
)


# -------------------------
# Column filter: remove constant columns
# -------------------------

def _validate_drop_constant_columns(params: Dict[str, Any]):
    require_no_params(params, "drop_constant_columns")


def _filter_drop_constant_columns(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = [col for col in ctx.df.columns if ctx.df[col].nunique(dropna=False) <= 1]
    return FilterResult(drop_columns=cols)


FILTER_REGISTRY.register(
    category="quality",
    name="drop_constant_columns",
    definition=FilterDefinition(
        fn=_filter_drop_constant_columns,
        validate_params=_validate_drop_constant_columns,
    ),
)
