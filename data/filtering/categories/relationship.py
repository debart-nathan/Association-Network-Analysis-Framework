from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np
import pandas as pd
from data.filtering.registry import FILTER_REGISTRY, FilterDefinition, FilterResult
from data.filtering.utils import (
    ensure_numeric_columns,
    require_list_param,
    require_number_param,
    require_no_params,
    validate_columns_exist,
)

if TYPE_CHECKING:
    from data.plan.engine import EngineContext

# -------------------------
# Column filter: drop highly correlated numeric columns
# -------------------------

def _validate_correlation(params: Dict[str, Any]):
    require_number_param(params, "threshold", "correlation")
    if "columns" in params:
        require_list_param(params, "columns", "correlation")


def _filter_correlation(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    threshold = require_number_param(params, "threshold", "correlation")
    numeric = ctx.df.select_dtypes(include="number")
    columns = params.get("columns")
    if columns is not None:
        columns = validate_columns_exist(ctx, columns, "correlation")
        numeric = ensure_numeric_columns(ctx, columns, "correlation")

    if numeric.shape[1] < 2:
        return FilterResult()

    corr = numeric.corr().abs()
    cols_to_drop = []
    for i, col in enumerate(corr.columns):
        for other in corr.columns[i + 1 :]:
            if corr.at[col, other] >= threshold:
                cols_to_drop.append(other)

    return FilterResult(drop_columns=list(dict.fromkeys(cols_to_drop)))


FILTER_REGISTRY.register(
    category="relationship",
    name="correlation",
    definition=FilterDefinition(
        fn=_filter_correlation,
        validate_params=_validate_correlation,
    ),
)


# -------------------------
# Column filter: drop columns with high variance inflation factor
# -------------------------

def _validate_multicollinearity(params: Dict[str, Any]):
    require_number_param(params, "threshold", "multicollinearity")
    if "columns" in params:
        require_list_param(params, "columns", "multicollinearity")


def _filter_multicollinearity(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    threshold = require_number_param(params, "threshold", "multicollinearity")
    columns = params.get("columns")
    numeric = ctx.df.select_dtypes(include="number")
    if columns is not None:
        columns = validate_columns_exist(ctx, columns, "multicollinearity")
        numeric = ensure_numeric_columns(ctx, columns, "multicollinearity")

    if numeric.shape[1] < 2:
        return FilterResult()

    values = numeric.dropna().to_numpy(dtype=float)
    if values.shape[0] == 0:
        return FilterResult()

    corr = np.corrcoef(values, rowvar=False)
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)

    vifs = np.diag(inv)
    cols_to_drop = [col for col, vif in zip(numeric.columns, vifs) if vif >= threshold]
    return FilterResult(drop_columns=cols_to_drop)


FILTER_REGISTRY.register(
    category="relationship",
    name="multicollinearity",
    definition=FilterDefinition(
        fn=_filter_multicollinearity,
        validate_params=_validate_multicollinearity,
    ),
)


# -------------------------
# Column filter: drop duplicate columns
# -------------------------

def _validate_drop_duplicate_columns(params: Dict[str, Any]):
    if "columns" in params:
        require_list_param(params, "columns", "drop_duplicate_columns")


def _filter_drop_duplicate_columns(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    columns = params.get("columns", list(ctx.df.columns))
    columns = validate_columns_exist(ctx, columns, "drop_duplicate_columns")
    seen = {}
    duplicates = []
    for col in columns:
        values = tuple(ctx.df[col].fillna("__NULL__").astype(str).tolist())
        if values in seen:
            duplicates.append(col)
        else:
            seen[values] = col
    return FilterResult(drop_columns=duplicates)


FILTER_REGISTRY.register(
    category="relationship",
    name="drop_duplicate_columns",
    definition=FilterDefinition(
        fn=_filter_drop_duplicate_columns,
        validate_params=_validate_drop_duplicate_columns,
    ),
)


# -------------------------
# Column filter: drop columns with constant ratio to another column
# -------------------------

def _validate_drop_constant_ratio_columns(params: Dict[str, Any]):
    if "columns" in params:
        require_list_param(params, "columns", "drop_constant_ratio_columns")
    if "tolerance" in params:
        require_number_param(params, "tolerance", "drop_constant_ratio_columns")


def _filter_drop_constant_ratio_columns(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    columns = params.get("columns", list(ctx.df.select_dtypes(include="number").columns))
    columns = validate_columns_exist(ctx, columns, "drop_constant_ratio_columns")
    tolerance = float(params.get("tolerance", 1e-6))
    numeric = ctx.df[columns].select_dtypes(include="number")
    cols_to_drop = set()

    for i, col_a in enumerate(numeric.columns):
        for col_b in numeric.columns[i + 1 :]:
            series_a = numeric[col_a].astype(float)
            series_b = numeric[col_b].astype(float)
            valid = series_a.notna() & series_b.notna() & (series_b != 0)
            if valid.sum() == 0:
                continue
            ratio = series_a[valid] / series_b[valid]
            if ratio.nunique() == 0:
                cols_to_drop.add(col_b)
            else:
                if ratio.max() - ratio.min() <= tolerance:
                    cols_to_drop.add(col_b)

    return FilterResult(drop_columns=list(cols_to_drop))


FILTER_REGISTRY.register(
    category="relationship",
    name="drop_constant_ratio_columns",
    definition=FilterDefinition(
        fn=_filter_drop_constant_ratio_columns,
        validate_params=_validate_drop_constant_ratio_columns,
    ),
)
