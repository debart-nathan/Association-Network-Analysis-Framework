import pandas as pd
import numpy as np
from sklearn import preprocessing as sk_preprocessing

from data.transformation.registry import (
    TRANSFORM_REGISTRY,
    TransformationDefinition,
    TransformationResult,
)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def one_hot_encoder_fn(ctx, inputs, params):
    col = inputs[0]
    series = ctx.df[col]

    dummy_na = params.get("dummy_na", False)

    encoded = pd.get_dummies(series, dummy_na=dummy_na)

    new_cols = {
        f"{col}_{c}": encoded[c]
        for c in encoded.columns
    }

    return TransformationResult(
        new_columns=new_cols,
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "encoding",
    TransformationDefinition(
        name="one_hot_encode",
        fn=one_hot_encoder_fn,
        allowed_params={"dummy_na": bool},
        description="Perform one-hot encoding on a categorical column.",
        is_derived=False,
        allowed_base=["categorical", "text", "boolean"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# ORDINAL ENCODING
# ============================================================

def ordinal_encoder_fn(ctx, inputs, params):
    col = inputs[0]
    series = ctx.df[col]

    # Handle NaN explicitly
    series_filled = series.fillna("__missing__")

    if "order" in params:
        order = params["order"]

        # Validate that all categories are covered
        unique_vals = set(series_filled.unique())
        missing = unique_vals - set(order) - {"__missing__"}
        if missing:
            raise ValueError(
                f"Ordinal encoding: categories {missing} not present in provided order."
            )

        categories = [order + ["__missing__"]]
        encoder = sk_preprocessing.OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        encoder = sk_preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    encoded = encoder.fit_transform(series_filled.values.reshape(-1, 1)).flatten()

    new_col_name = f"{col}_ordinal"
    new_col = pd.Series(encoded, index=series.index, name=new_col_name)

    return TransformationResult(
        new_columns={new_col_name: new_col},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "encoding",
    TransformationDefinition(
        name="ordinal_encode",
        fn=ordinal_encoder_fn,
        allowed_params={"order": list},
        description="Perform ordinal encoding on a categorical column. "
                    "If 'order' is provided, it defines the category order.",
        is_derived=False,
        allowed_base=["categorical", "text", "boolean"],
        output_schema=("numeric", "ordinal"),
    )
)


# ============================================================
# LABEL ENCODING
# ============================================================

def label_encoder_fn(ctx, inputs, params):
    col = inputs[0]
    series = ctx.df[col]

    # Replace NaN with explicit category
    series_filled = series.fillna("__missing__")

    encoder = sk_preprocessing.LabelEncoder()
    encoded = encoder.fit_transform(series_filled)

    new_col_name = f"{col}_label"
    encoded = np.asarray(encoded)
    new_col = pd.Series(encoded, index=ctx.df.index, name=new_col_name)


    return TransformationResult(
        new_columns={new_col_name: new_col},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "encoding",
    TransformationDefinition(
        name="label_encode",
        fn=label_encoder_fn,
        allowed_params={},
        description="Perform label encoding on a categorical column. "
                    "Produces arbitrary integer codes with no ordinal meaning.",
        is_derived=False,
        allowed_base=["categorical", "text", "boolean"],
        output_schema=("numeric", "discrete"),
    )
)
