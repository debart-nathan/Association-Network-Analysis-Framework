import pandas as pd
import numpy as np

from data.transformation.registry import (
    TRANSFORM_REGISTRY,
    TransformationDefinition,
    TransformationResult,
)

# ============================================================
# NUMERIC + NUMERIC INTERACTIONS
# ============================================================

def add_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1, s2 = ctx.df[col1], ctx.df[col2]

    col_name = f"{col1}_plus_{col2}"
    new_col = pd.Series(s1 + s2, index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="add_columns",
        fn=add_columns_fn,
        allowed_params={},
        description="Add two numeric columns.",
        is_derived=True,
        allowed_base=[("numeric", "numeric")],
        output_schema=("numeric", "continuous"),
    )
)


def multiply_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1, s2 = ctx.df[col1], ctx.df[col2]

    col_name = f"{col1}_times_{col2}"
    new_col = pd.Series(s1 * s2, index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="multiply_columns",
        fn=multiply_columns_fn,
        allowed_params={},
        description="Multiply two numeric columns.",
        is_derived=True,
        allowed_base=[("numeric", "numeric")],
        output_schema=("numeric", "continuous"),
    )
)


def subtract_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1, s2 = ctx.df[col1], ctx.df[col2]

    col_name = f"{col1}_minus_{col2}"
    new_col = pd.Series(s1 - s2, index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="subtract_columns",
        fn=subtract_columns_fn,
        allowed_params={},
        description="Subtract two numeric columns.",
        is_derived=True,
        allowed_base=[("numeric", "numeric")],
        output_schema=("numeric", "continuous"),
    )
)


def divide_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1 = ctx.df[col1]
    s2 = ctx.df[col2].replace(0, pd.NA)

    col_name = f"{col1}_div_{col2}"
    new_col = pd.Series(s1 / s2, index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="divide_columns",
        fn=divide_columns_fn,
        allowed_params={},
        description="Divide two numeric columns safely.",
        is_derived=True,
        allowed_base=[("numeric", "numeric")],
        output_schema=("numeric", "continuous"),
    )
)


def ratio_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1 = ctx.df[col1]
    s2 = ctx.df[col2].replace(0, pd.NA)

    col_name = f"{col1}_ratio_{col2}"
    new_col = pd.Series(s1 / s2, index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="ratio_columns",
        fn=ratio_columns_fn,
        allowed_params={},
        description="Compute ratio of two numeric columns.",
        is_derived=True,
        allowed_base=[("numeric", "numeric")],
        output_schema=("numeric", "continuous"),
    )
)


def log_ratio_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1 = ctx.df[col1].replace(0, pd.NA)
    s2 = ctx.df[col2].replace(0, pd.NA)

    col_name = f"log_{col1}_over_{col2}"
    new_col = pd.Series(np.log(s1 / s2), index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="log_ratio_columns",
        fn=log_ratio_columns_fn,
        allowed_params={},
        description="Compute log ratio of two numeric columns.",
        is_derived=True,
        allowed_base=[("numeric", "numeric")],
        output_schema=("numeric", "continuous"),
    )
)


# ============================================================
# POLYNOMIAL FEATURES (CORRECT VERSION)
# ============================================================

def polynomial_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    degree = params.get("degree", 2)

    s1 = ctx.df[col1]
    s2 = ctx.df[col2]

    # True polynomial interaction: x^d, y^d, x^(d-1)*y, ..., x*y^(d-1)
    new_cols = {}
    for i in range(degree + 1):
        term_name = f"{col1}^{degree-i}_{col2}^{i}"
        values = (s1 ** (degree - i)) * (s2 ** i)
        new_cols[term_name] = pd.Series(values, index=s1.index, name=term_name)

    return TransformationResult(new_cols, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="polynomial_columns",
        fn=polynomial_columns_fn,
        allowed_params={"degree": int},
        description="Generate polynomial interaction terms.",
        is_derived=True,
        allowed_base=[("numeric", "numeric")],
        output_schema=("numeric", "continuous"),
    )
)


# ============================================================
# CATEGORICAL + CATEGORICAL INTERACTIONS
# ============================================================

def cross_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1 = ctx.df[col1].astype(str)
    s2 = ctx.df[col2].astype(str)

    col_name = f"{col1}_cross_{col2}"
    new_col = pd.Series(s1 + "_" + s2, index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="cross_columns",
        fn=cross_columns_fn,
        allowed_params={},
        description="Cross two categorical columns.",
        is_derived=True,
        allowed_base=[("categorical", "categorical")],
        output_schema=("categorical", "nominal"),
    )
)


# ============================================================
# TEXT + TEXT INTERACTIONS
# ============================================================

def combine_text_columns_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1 = ctx.df[col1].fillna("").astype(str)
    s2 = ctx.df[col2].fillna("").astype(str)

    col_name = f"{col1}_combined_{col2}"
    new_col = pd.Series(s1 + " " + s2, index=s1.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="combine_text_columns",
        fn=combine_text_columns_fn,
        allowed_params={},
        description="Combine two text columns.",
        is_derived=True,
        allowed_base=[("text", "text")],
        output_schema=("text", "short_text"),
    )
)


# ============================================================
# NUMERIC + CATEGORICAL INTERACTIONS
# ============================================================

def combine_numeric_categorical_fn(ctx, inputs, params):
    num_col, cat_col = inputs
    s_num = ctx.df[num_col].fillna(0).astype(str)
    s_cat = ctx.df[cat_col].fillna("missing").astype(str)

    col_name = f"{num_col}_combined_{cat_col}"
    new_col = pd.Series(s_num + "_" + s_cat, index=s_num.index, name=col_name)

    return TransformationResult({col_name: new_col}, terminal=True)


TRANSFORM_REGISTRY.register(
    "interactions",
    TransformationDefinition(
        name="combine_numeric_categorical",
        fn=combine_numeric_categorical_fn,
        allowed_params={},
        description="Combine numeric and categorical columns.",
        is_derived=True,
        allowed_base=[("numeric", "categorical"), ("categorical", "numeric")],
        output_schema=("categorical", "nominal"),
    )
)
