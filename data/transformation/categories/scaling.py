import pandas as pd
import numpy as np
from data.transformation.registry import (
    TransformationResult,
    TransformationDefinition,
    TRANSFORM_REGISTRY,
)
from scipy.stats import boxcox, yeojohnson
# -----------------------------
# Z-SCORE NORMALIZATION
# -----------------------------

def zscore_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    # Use population std (ddof=0) for ML consistency
    std = s.std(ddof=0)

    if std == 0:
        # Constant column → all zeros
        zscores = pd.Series(0, index=s.index)
    else:
        zscores = (s - s.mean()) / std

    col_name = f"{col}_zscore"
    new_col = pd.Series(zscores, index=s.index, name=col_name)

    return TransformationResult(
        new_columns={col_name: new_col},
        terminal=True,
    )

TRANSFORM_REGISTRY.register(
    "scaling",
    TransformationDefinition(
        name="zscore",
        fn=zscore_fn,
        allowed_params={},
        description="Scale a numerical column using z-score normalization.",
        is_derived=False,
        allowed_base=["numerical"],
        output_schema=("numeric", "continuous"),
    )
)

# -----------------------------
# MIN-MAX NORMALIZATION
# -----------------------------

def minmax_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    # Output range (defaults to 0–1)
    out_min = params.get("min", 0.0)
    out_max = params.get("max", 1.0)

    # Normalization range (defaults to data min/max)
    data_min = params.get("data_min", s.min())
    data_max = params.get("data_max", s.max())

    col_name = f"{col}_minmax"

    # Avoid division by zero when column is constant
    range_ = data_max - data_min
    if range_ == 0:
        # Entire column is constant → return midpoint of output range
        scaled = pd.Series(
            (out_min + out_max) / 2,
            index=s.index,
            name=col_name,
        )
    else:
        scaled = ((s - data_min) / range_) * (out_max - out_min) + out_min
        scaled.name = col_name

    return TransformationResult(
        new_columns={col_name: scaled},
        terminal=True,
    )

TRANSFORM_REGISTRY.register(
    "scaling",
    TransformationDefinition(
        name="minmax",
        fn=minmax_fn,
        allowed_params={
            "min": float,
            "max": float,
            "data_min": float,
            "data_max": float,
        },
        description="Scale a numerical column using min-max normalization.",
        is_derived=False,
        allowed_base=["numerical"],
        output_schema=("numeric", "continuous"),
    )
)

# -----------------------------
# RANK TRANSFORM
# -----------------------------

def rank_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    method = params.get("method", "average")
    ascending = params.get("ascending", True)

    ranked = s.rank(method=method, ascending=ascending)

    col_name = f"{col}_rank"
    new_col = pd.Series(ranked, index=s.index, name=col_name)

    return TransformationResult(
        new_columns={col_name: new_col},
        terminal=True,
    )

TRANSFORM_REGISTRY.register(
    "scaling",
    TransformationDefinition(
        name="rank",
        fn=rank_fn,
        allowed_params={
            "method": str,
            "ascending": bool,
        },
        description="Rank a numerical column, with options for method and order.",
        is_derived=False,
        allowed_base=["numerical"],
        output_schema=("numeric", "continuous"),
    )
)

# -----------------------------
# Box-Cox
# -----------------------------


def boxcox_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    if (s <= 0).any():
        raise ValueError("Box-Cox transform requires strictly positive values")

    if s.isna().any():
        raise ValueError("Box-Cox requires no missing values; impute first")

    if s.nunique() == 1:
        raise ValueError("Box-Cox undefined for constant columns")

    lam = params.get("lambda", None)

    if lam is None:
        result = boxcox(s)
    else:
        result = boxcox(s, lmbda=lam)

    if isinstance(result, tuple):
        transformed_array, lam = result
    else:
        transformed_array = result

    col_name = f"{col}_boxcox"
    new_col = pd.Series(transformed_array, index=s.index, name=col_name)

    return TransformationResult(
        new_columns={col_name: new_col},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "scaling",
    TransformationDefinition(
        name="boxcox",
        fn=boxcox_fn,
        allowed_params={
            "lambda": (int, float),
        },
        description="Apply Box-Cox power transform to a numerical column (fits lambda if not provided).",
        is_derived=False,
        allowed_base=["numerical"],
        output_schema=("numeric", "continuous"),
    )
)

# -----------------------------
# Yeo-Johnson
# -----------------------------

def yeojohnson_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    if s.isna().any():
        raise ValueError("Yeo-Johnson requires no missing values; impute first")

    if s.nunique() == 1:
        raise ValueError("Yeo-Johnson undefined for constant columns")

    lam = params.get("lambda", None)

    # Call once – this is the real source
    if lam is None:
        result = yeojohnson(s)
    else:
        result = yeojohnson(s, lmbda=lam)

    # Normalize shape: after this, transformed_array is never a tuple
    if isinstance(result, tuple):
        transformed_array, lam = result
    else:
        transformed_array = result

    transformed_array = np.asarray(transformed_array)

    col_name = f"{col}_yeojohnson"
    new_col = pd.Series(transformed_array, index=s.index, name=col_name)

    return TransformationResult(
        new_columns={col_name: new_col},
        terminal=True,
    )



TRANSFORM_REGISTRY.register(
    "scaling",
    TransformationDefinition(
        name="yeojohnson",
        fn=yeojohnson_fn,
        allowed_params={
            "lambda": (int, float),
        },
        description="Apply Yeo-Johnson power transform to a numerical column (fits lambda if not provided).",
        is_derived=False,
        allowed_base=["numerical"],
        output_schema=("numeric", "continuous"),
    )
)
