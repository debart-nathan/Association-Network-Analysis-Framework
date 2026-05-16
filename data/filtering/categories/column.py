from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List
from data.filtering.registry import FILTER_REGISTRY, FilterDefinition, FilterResult
from data.filtering.utils import (
    drop_other_columns,
    require_list_param,
    require_no_params,
    require_param,
    validate_columns_exist,
)

if TYPE_CHECKING:
    from data.plan.engine import EngineContext

# name‑based column filtering

# -------------------------
# Column filter: drop columns by name
# -------------------------

def _validate_drop_by_name(params: Dict[str, Any]):
    require_list_param(params, "columns", "drop_by_name")


def _filter_drop_by_name(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "drop_by_name"),
        "drop_by_name",
    )
    return FilterResult(drop_columns=cols)

FILTER_REGISTRY.register(
    category="column",
    name="drop_by_name",
    definition=FilterDefinition(
        fn=_filter_drop_by_name,
        validate_params=_validate_drop_by_name,
    ),
)


# -------------------------
# Column filter: keep only named columns
# -------------------------

def _validate_keep_only(params: Dict[str, Any]):
    require_list_param(params, "columns", "keep_only")


def _filter_keep_only(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "keep_only"),
        "keep_only",
    )
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="column",
    name="keep_only",
    definition=FilterDefinition(
        fn=_filter_keep_only,
        validate_params=_validate_keep_only,
    ),
)


# -------------------------
# Column filter: drop only named columns
# -------------------------

def _validate_drop_only(params: Dict[str, Any]):
    require_list_param(params, "columns", "drop_only")


def _filter_drop_only(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    cols = validate_columns_exist(
        ctx,
        require_list_param(params, "columns", "drop_only"),
        "drop_only",
    )
    return FilterResult(drop_columns=cols)


FILTER_REGISTRY.register(
    category="column",
    name="drop_only",
    definition=FilterDefinition(
        fn=_filter_drop_only,
        validate_params=_validate_drop_only,
    ),
)


# -------------------------
# Column filter: drop columns by prefix
# -------------------------

def _validate_prefix(params: Dict[str, Any]):
    require_param(params, "prefix", "drop_by_prefix")


def _filter_drop_by_prefix(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    prefix = require_param(params, "prefix", "drop_by_prefix")
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


# -------------------------
# Column filter: keep columns by prefix
# -------------------------

def _validate_keep_by_prefix(params: Dict[str, Any]):
    require_param(params, "prefix", "keep_by_prefix")


def _filter_keep_by_prefix(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    prefix = require_param(params, "prefix", "keep_by_prefix")
    cols = [c for c in ctx.df.columns if c.startswith(prefix)]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="column",
    name="keep_by_prefix",
    definition=FilterDefinition(
        fn=_filter_keep_by_prefix,
        validate_params=_validate_keep_by_prefix,
    ),
)


# -------------------------
# Column filter: drop columns by suffix
# -------------------------

def _validate_suffix(params: Dict[str, Any]):
    require_param(params, "suffix", "drop_by_suffix")


def _filter_drop_by_suffix(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    suffix = require_param(params, "suffix", "drop_by_suffix")
    cols = [c for c in ctx.df.columns if c.endswith(suffix)]
    return FilterResult(drop_columns=cols)

FILTER_REGISTRY.register(
    category="column",
    name="drop_by_suffix",
    definition=FilterDefinition(
        fn=_filter_drop_by_suffix,
        validate_params=_validate_suffix,
    ),
)


# -------------------------
# Column filter: keep columns by suffix
# -------------------------

def _validate_keep_by_suffix(params: Dict[str, Any]):
    require_param(params, "suffix", "keep_by_suffix")


def _filter_keep_by_suffix(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    suffix = require_param(params, "suffix", "keep_by_suffix")
    cols = [c for c in ctx.df.columns if c.endswith(suffix)]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="column",
    name="keep_by_suffix",
    definition=FilterDefinition(
        fn=_filter_keep_by_suffix,
        validate_params=_validate_keep_by_suffix,
    ),
)


# -------------------------
# Column filter: drop columns by regex
# -------------------------

import re

def _validate_regex(params: Dict[str, Any]):
    require_param(params, "pattern", "drop_by_regex")

def _filter_drop_by_regex(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    pattern = re.compile(require_param(params, "pattern", "drop_by_regex"))
    cols = [c for c in ctx.df.columns if pattern.search(c)]
    return FilterResult(drop_columns=cols)

FILTER_REGISTRY.register(
    category="column",
    name="drop_by_regex",
    definition=FilterDefinition(
        fn=_filter_drop_by_regex,
        validate_params=_validate_regex,
    ),
)


# -------------------------
# Column filter: keep columns by regex
# -------------------------

def _validate_keep_by_regex(params: Dict[str, Any]):
    require_param(params, "pattern", "keep_by_regex")


def _filter_keep_by_regex(ctx: EngineContext, inputs: List[str], params: Dict[str, Any]) -> FilterResult:
    pattern = re.compile(require_param(params, "pattern", "keep_by_regex"))
    cols = [c for c in ctx.df.columns if pattern.search(c)]
    return FilterResult(drop_columns=drop_other_columns(ctx, cols))


FILTER_REGISTRY.register(
    category="column",
    name="keep_by_regex",
    definition=FilterDefinition(
        fn=_filter_keep_by_regex,
        validate_params=_validate_keep_by_regex,
    ),
)

