import pandas as pd
import numpy as np
from sklearn import preprocessing as sk_preprocessing

from data.transformation.registry import (
    TRANSFORM_REGISTRY,
    TransformationDefinition,
    TransformationResult,
)


def one_hot_encoder_fn(ctx, inputs, params):
    # One-hot encoding always takes exactly one input column
    col = inputs[0]
    series = ctx.df[col]

    encoded = pd.get_dummies(series)

    new_cols = {f"{col}_{c}": encoded[c] for c in encoded.columns}

    return TransformationResult(
        new_columns=new_cols,
        terminal=True
    )


TRANSFORM_REGISTRY.register(
    "encoding",
    TransformationDefinition(
        name="one_hot_encode",
        fn=one_hot_encoder_fn,
        allowed_params={},
        description="Perform one-hot encoding on a categorical column.",
        is_derived=False,
        allowed_base=["categorical", "text", "boolean"],
    )
)

def ordinal_encoder_fn(ctx, inputs, params):
    col = inputs[0]
    series = ctx.df[col]

    # If user provides an explicit order
    if "order" in params:
        categories = [params["order"]]
        encoder = sk_preprocessing.OrdinalEncoder(categories=categories)
    else:
        encoder = sk_preprocessing.OrdinalEncoder()

    encoded = np.asarray(
        encoder.fit_transform(series.values.reshape(-1, 1)).flatten()
    )


    new_col_name = f"{col}_ordinal"
    new_col = pd.Series(encoded, name=new_col_name)

    return TransformationResult(
        new_columns={new_col_name: new_col},
        terminal=True
    )



TRANSFORM_REGISTRY.register(
    "encoding",
    TransformationDefinition(
        name="ordinal_encode",
        fn=ordinal_encoder_fn,
        allowed_params={"order": list},
        description="Perform ordinal encoding on a categorical column.",
        is_derived=False,
        allowed_base=["categorical", "text", "boolean"],
    )
)

def label_encoder_fn(ctx, inputs, params):
    col = inputs[0]
    series = ctx.df[col]

    encoder = sk_preprocessing.LabelEncoder()
    encoded = np.asarray(encoder.fit_transform(series))

    new_col_name = f"{col}_label"
    new_col = pd.Series(encoded, name=new_col_name)

    return TransformationResult(
        new_columns={new_col_name: new_col},
        terminal=True
    )

TRANSFORM_REGISTRY.register(
    "encoding",
    TransformationDefinition(
        name="label_encode",
        fn=label_encoder_fn,
        allowed_params={},
        description="Perform label encoding on a categorical column.",
        is_derived=False,
        allowed_base=["categorical", "text", "boolean"],
    )
)