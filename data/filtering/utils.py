from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:
    from data.plan.engine import EngineContext

import pandas as pd


def require_param(params: Dict[str, Any], key: str, filter_name: str) -> Any:
    if key not in params:
        raise ValueError(f"{filter_name} requires param '{key}'.")
    return params[key]


def require_list_param(params: Dict[str, Any], key: str, filter_name: str) -> List[Any]:
    value = require_param(params, key, filter_name)
    if not isinstance(value, list):
        raise ValueError(f"{filter_name} '{key}' must be a list.")
    return value


def require_number_param(params: Dict[str, Any], key: str, filter_name: str) -> float:
    value = require_param(params, key, filter_name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{filter_name} '{key}' must be a number.")
    return float(value)


def require_int_param(params: Dict[str, Any], key: str, filter_name: str) -> int:
    value = require_param(params, key, filter_name)
    if not isinstance(value, int):
        raise ValueError(f"{filter_name} '{key}' must be an integer.")
    return value


def require_non_negative_int_param(params: Dict[str, Any], key: str, filter_name: str) -> int:
    value = require_int_param(params, key, filter_name)
    if value < 0:
        raise ValueError(f"{filter_name} '{key}' must be non-negative.")
    return value


def require_no_params(params: Dict[str, Any], filter_name: str) -> None:
    if params:
        raise ValueError(f"{filter_name} takes no parameters.")


def reject_unknown_params(params: Dict[str, Any], allowed: Iterable[str], filter_name: str) -> None:
    invalid_keys = set(params) - set(allowed)
    if invalid_keys:
        raise ValueError(
            f"{filter_name} accepts only params {sorted(allowed)}. Got: {sorted(invalid_keys)}"
        )


def require_at_least_one_param(params: Dict[str, Any], keys: Sequence[str], filter_name: str) -> None:
    if not any(key in params for key in keys):
        raise ValueError(f"{filter_name} requires one of {list(keys)}.")


def validate_columns_exist(ctx: EngineContext, columns: List[str], filter_name: str) -> List[str]:
    missing = [c for c in columns if c not in ctx.df.columns]
    if missing:
        raise RuntimeError(f"{filter_name} requested unknown columns: {missing}")
    return columns


def require_threshold_and_columns(
    params: Dict[str, Any],
    filter_name: str,
    threshold_key: str = "threshold",
) -> tuple[List[Any], float]:
    columns = require_list_param(params, "columns", filter_name)
    threshold = require_number_param(params, threshold_key, filter_name)
    return columns, threshold


def ensure_numeric_columns(
    ctx: EngineContext,
    columns: List[str],
    filter_name: str,
) -> pd.DataFrame:
    validate_columns_exist(ctx, columns, filter_name)
    numeric = ctx.df.select_dtypes(include="number")
    non_numeric = [c for c in columns if c not in numeric.columns]
    if non_numeric:
        raise RuntimeError(
            f"{filter_name} requested non-numeric columns: {non_numeric}"
        )
    return numeric[columns]


def drop_other_columns(ctx: EngineContext, keep_columns: Iterable[str]) -> List[str]:
    keep_set = set(keep_columns)
    return [col for col in ctx.df.columns if col not in keep_set]


def select_extreme_columns(series: pd.Series, top_k: int | None, bottom_k: int | None) -> List[str]:
    selected: List[str] = []

    if top_k is not None and top_k > 0:
        selected.extend(series.nlargest(top_k).index.tolist())

    if bottom_k is not None and bottom_k > 0:
        selected.extend(series.nsmallest(bottom_k).index.tolist())

    return list(dict.fromkeys(selected))


def require_selection_mode(params: Dict[str, Any], filter_name: str) -> str:
    mode = params.get("mode", "remove")
    if mode not in ("keep", "remove"):
        raise ValueError(f"{filter_name} mode must be 'keep' or 'remove'.")
    return mode


def apply_selection_mode(columns: List[str], selected: List[str], mode: str) -> List[str]:
    if mode == "remove":
        return selected
    if mode == "keep":
        keep_set = set(selected)
        return [col for col in columns if col not in keep_set]
    raise ValueError("mode must be 'keep' or 'remove'.")
