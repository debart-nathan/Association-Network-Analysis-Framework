from __future__ import annotations

# variance, cardinality, distribution

from typing import TYPE_CHECKING, Any, Dict, List
import pandas as pd
from data.filtering.registry import FILTER_REGISTRY, FilterDefinition, FilterResult
from data.filtering.utils import (
    apply_selection_mode,
    require_at_least_one_param,
    require_list_param,
    require_non_negative_int_param,
    require_number_param,
    require_param,
    require_selection_mode,
    select_extreme_columns,
    validate_columns_exist,
)

if TYPE_CHECKING:
    from data.plan.engine import EngineContext

# -----------------------
# drop columns with low/high variance
# --------------------------

def _validate_variance(params: Dict[str, Any]):
    require_list_param(params, "columns", "variance")
    require_at_least_one_param(params, ("min_threshold", "max_threshold"), "variance")


def _filter_variance(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    columns = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "variance"),
        "variance",
    )
    min_t = params.get("min_threshold")
    max_t = params.get("max_threshold")
    variances = ctx.df[columns].var(numeric_only=True)
    mask = pd.Series(False, index=variances.index)
    if min_t is not None:
        mask |= variances < float(min_t)
    if max_t is not None:
        mask |= variances > float(max_t)
    cols_to_drop = variances[mask].index.intersection(columns).tolist()
    return FilterResult(drop_columns=cols_to_drop)

FILTER_REGISTRY.register(
    category="distribution",
    name="variance",
    definition=FilterDefinition(
        fn=_filter_variance,
        validate_params=_validate_variance,
    ),
)

# -----------------------
# drop rare/frequent values for columns
# --------------------------

def _validate_cardinality(params: Dict[str, Any]):
    require_list_param(params, "columns", "cardinality")
    require_at_least_one_param(params, ("min_threshold", "max_threshold"), "cardinality")


def _filter_cardinality(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    columns = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "cardinality"),
        "cardinality",
    )
    min_t = params.get("min_threshold")
    max_t = params.get("max_threshold")

    cardinalities = ctx.df[columns].nunique().astype(float)
    mask = pd.Series(False, index=cardinalities.index)

    if min_t is not None:
        mask |= cardinalities < float(min_t)
    if max_t is not None:
        mask |= cardinalities > float(max_t)

    cols_to_drop = cardinalities[mask].index.tolist()

    return FilterResult(drop_columns=cols_to_drop)


FILTER_REGISTRY.register(
    category="distribution",
    name="cardinality",
    definition=FilterDefinition(
        fn=_filter_cardinality,
        validate_params=_validate_cardinality,
    ),
)


# -----------------------
# drop columns with top/bottom k variance
# --------------------------

def _validate_variance_k(params: Dict[str, Any]):
    require_list_param(params, "columns", "variance_k")
    require_at_least_one_param(params, ("top_k", "bottom_k"), "variance_k")
    if "top_k" in params:
        require_non_negative_int_param(params, "top_k", "variance_k")
    if "bottom_k" in params:
        require_non_negative_int_param(params, "bottom_k", "variance_k")
    require_selection_mode(params, "variance_k")


def _filter_variance_k(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    columns = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "variance_k"),
        "variance_k",
    )
    top_k = params.get("top_k")
    bottom_k = params.get("bottom_k")
    mode = params.get("mode", "remove")

    variances = ctx.df[columns].var(numeric_only=True).dropna()
    selected = select_extreme_columns(variances, top_k, bottom_k)
    cols_to_drop = apply_selection_mode(columns, selected, mode)

    return FilterResult(drop_columns=cols_to_drop)


FILTER_REGISTRY.register(
    category="distribution",
    name="variance_k",
    definition=FilterDefinition(
        fn=_filter_variance_k,
        validate_params=_validate_variance_k,
    ),
)


# -----------------------
# drop columns with top/bottom k cardinality
# --------------------------

def _validate_cardinality_k(params: Dict[str, Any]):
    require_list_param(params, "columns", "cardinality_k")
    require_at_least_one_param(params, ("top_k", "bottom_k"), "cardinality_k")
    if "top_k" in params:
        require_non_negative_int_param(params, "top_k", "cardinality_k")
    if "bottom_k" in params:
        require_non_negative_int_param(params, "bottom_k", "cardinality_k")
    require_selection_mode(params, "cardinality_k")


def _filter_cardinality_k(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    columns = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "cardinality_k"),
        "cardinality_k",
    )
    top_k = params.get("top_k")
    bottom_k = params.get("bottom_k")
    mode = params.get("mode", "remove")

    cardinalities = ctx.df[columns].nunique().dropna()
    selected = select_extreme_columns(cardinalities, top_k, bottom_k)
    cols_to_drop = apply_selection_mode(columns, selected, mode)

    return FilterResult(drop_columns=cols_to_drop)

FILTER_REGISTRY.register(
    category="distribution",
    name="cardinality_k",
     definition=FilterDefinition(
        fn=_filter_cardinality_k,
        validate_params=_validate_cardinality_k,
    ),
)