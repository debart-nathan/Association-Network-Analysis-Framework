from data.transformation.registry import TRANSFORM_REGISTRY, TransformationDefinition, TransformationResult
from data.schema.schema_types import SchemaEntry

import pandas as pd
from sklearn.impute import KNNImputer


# ============================================================
# DROP EMPTY
# ============================================================

def drop_empty_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    treat_empty_string = params.get("treat_empty_string", True)
    if treat_empty_string:
        mask = ~(s.isna() | (s == ""))
    else:
        mask = ~s.isna()

    new_col = pd.Series(mask.astype(int), index=s.index, name=f"{col}_not_empty")

    return TransformationResult(
        new_columns={f"{col}_not_empty": new_col},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="drop_empty",
        fn=drop_empty_fn,
        allowed_params={"treat_empty_string": bool},
        description="Drop rows where the column is empty or NaN.",
        is_derived=True,
    )
)


# ============================================================
# IMPUTE MEAN
# ============================================================

def impute_mean_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].fillna(ctx.df[col].mean())

    return TransformationResult(
        new_columns={f"{col}_imputed_mean": s}
    )


TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_mean",
        fn=impute_mean_fn,
        allowed_params={},
        description="Fill missing values with the column mean.",
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
        is_derived=True,
    )
)


# ============================================================
# IMPUTE MEDIAN
# ============================================================

def impute_median_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].fillna(ctx.df[col].median())

    return TransformationResult(
        new_columns={f"{col}_imputed_median": s}
    )


TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_median",
        fn=impute_median_fn,
        allowed_params={},
        description="Fill missing values with the column median.",
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
        is_derived=True,
    )
)


# ============================================================
# IMPUTE MODE
# ============================================================

def impute_mode_fn(ctx, inputs, params):
    col = inputs[0]
    mode_value = ctx.df[col].mode().iloc[0]
    s = ctx.df[col].fillna(mode_value)

    return TransformationResult(
        new_columns={f"{col}_imputed_mode": s}
    )


TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_mode",
        fn=impute_mode_fn,
        allowed_params={},
        description="Fill missing values with the column mode.",
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
        is_derived=True,
    )
)


# ============================================================
# IMPUTE KNN
# ============================================================

def impute_knn_fn(ctx, inputs, params):
    features = params.get("features")
    target = params.get("target")

    if not features or not target:
        raise ValueError("KNN imputation requires 'features' and 'target'.")

    df = ctx.df[features + [target]]
    imputer = KNNImputer(n_neighbors=params.get("n_neighbors", 5))

    imputed = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed, columns=df.columns, index=df.index)

    new_col = f"{target}_imputed_knn"

    return TransformationResult(
        new_columns={new_col: imputed_df[target]}
    )


TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_knn",
        fn=impute_knn_fn,
        allowed_params={"n_neighbors": int, "features": list, "target": str},
        description="Fill missing values using KNN imputation.",
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
        is_derived=True,
    )
)


# ============================================================
# IMPUTE CONSTANT
# ============================================================

def impute_constant_fn(ctx, inputs, params):
    col = inputs[0]
    value = params.get("constant_value", 0)
    s = ctx.df[col].fillna(value)

    return TransformationResult(
        new_columns={f"{col}_imputed_constant": s}
    )


TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_constant",
        fn=impute_constant_fn,
        allowed_params={"constant_value": object},
        description="Fill missing values with a constant value.",
        is_derived=True,
    )
)
