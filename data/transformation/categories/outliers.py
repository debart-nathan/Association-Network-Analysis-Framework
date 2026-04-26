import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from data.transformation.registry import (
    TransformationResult,
    TransformationDefinition,
    TRANSFORM_REGISTRY,
)

def winsorize_fn(ctx, inputs, params):
    col = inputs[0]

    lower = params.get("lower_limit", 0.05)
    upper = params.get("upper_limit", 0.05)

    if not (0 <= lower <= 1 and 0 <= upper <= 1):
        raise ValueError("winsorize limits must be between 0 and 1")

    s = ctx.df[col]
    wins = winsorize(s.to_numpy(), limits=(lower, upper))

    col_name = f"{col}_winsorized"
    new_col = pd.Series(wins, index=s.index, name=col_name)

    return TransformationResult(
        new_columns={col_name: new_col},
        terminal=True,
    )

TRANSFORM_REGISTRY.register(
    "outliers",
    TransformationDefinition(
        name="winsorize",
        fn=winsorize_fn,
        allowed_params={"lower_limit": float, "upper_limit": float},
        description="Perform winsorization on a numerical column.",
        is_derived=False,
        allowed_base=["numerical"],
    )
)

def clipping_fn(ctx, inputs, params):
    col = inputs[0]

    lower = params.get("lower_limit", None)
    upper = params.get("upper_limit", None)

    s = ctx.df[col]
    clipped = s.clip(lower=lower, upper=upper)

    col_name = f"{col}_clipped"
    new_col = pd.Series(clipped, index=s.index, name=col_name)

    return TransformationResult(
        new_columns={col_name: new_col},
        terminal=True,
    )

TRANSFORM_REGISTRY.register(
    "outliers",
    TransformationDefinition(
        name="clip",
        fn=clipping_fn,
        allowed_params={"lower_limit": float, "upper_limit": float},
        description="Perform clipping on a numerical column.",
        is_derived=False,
        allowed_base=["numerical"],
    )
)

def log_transform_fn(ctx, inputs, params):
    col = inputs[0]

    s = ctx.df[col]
    # Optional offset to avoid log(0) or log(negative)
    offset = params.get("offset", 0)

    arr = s.to_numpy() + offset

    if (arr <= 0).any():
        raise ValueError("Log transform received non-positive values after offset.")

    transformed = np.log(arr)

    col_name = f"{col}_log"
    new_col = pd.Series(transformed, index=s.index, name=col_name)

    return TransformationResult(
        new_columns={col_name: new_col},
        terminal=True,
    )

TRANSFORM_REGISTRY.register(
    "outliers",
    TransformationDefinition(
        name="log_transform",
        fn=log_transform_fn,
        allowed_params={},
        description="Apply log transform to compress large values and reduce outlier influence.",
        is_derived=False,
        allowed_base=["numerical"],
    )
)
