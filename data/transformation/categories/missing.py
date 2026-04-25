from data.transformation.registry import TRANSFORM_REGISTRY, TransformationDefinition
from data.schema.schema_types import SchemaEntry
import pandas as pd
from sklearn.impute import KNNImputer


def drop_empty_fn(data: pd.Series | pd.DataFrame,
                  schema: SchemaEntry | None,
                  params: dict,
                  context: dict) -> pd.Series | pd.DataFrame:

    treat_empty_string = params.get("treat_empty_string", True)

    if isinstance(data, pd.Series):
        s = data
        if treat_empty_string:
            s = s.replace("", pd.NA)
        return s.dropna()

    # If DataFrame (rare for column-level transforms)
    df = data.copy()
    if treat_empty_string:
        df = df.replace("", pd.NA)
    return df.dropna()



TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="drop_empty",
        fn=drop_empty_fn,
        allowed_params={"treat_empty_string": bool},
        description="Drop rows where the column is empty or NaN."
    )
)



def impute_mean_fn(data: pd.Series | pd.DataFrame,
                  schema: SchemaEntry | None,
                  params: dict,
                  context: dict) -> pd.Series | pd.DataFrame:
    return data.fillna(data.mean())

TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_mean",
        fn=impute_mean_fn,
        allowed_params={},  # no params needed
        description="Fill missing values with the column mean.",

        # Compatibility rules
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
    )
)


def impute_median_fn(data: pd.Series | pd.DataFrame,
                    schema: SchemaEntry | None,
                    params: dict,
                    context: dict) -> pd.Series | pd.DataFrame:
    return data.fillna(data.median())

TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_median",
        fn=impute_median_fn,
        allowed_params={},  # no params needed
        description="Fill missing values with the column median.",

        # Compatibility rules
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
    )
)

def impute_mode_fn(data: pd.Series | pd.DataFrame,
                  schema: SchemaEntry | None,
                  params: dict,
                  context: dict) -> pd.Series | pd.DataFrame:
    return data.fillna(data.mode().iloc[0])

TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_mode",
        fn=impute_mode_fn,
        allowed_params={},  # no params needed
        description="Fill missing values with the column mode.",

        # Compatibility rules
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
    )
)


def impute_knn_fn(data: pd.Series | pd.DataFrame,
                 schema: SchemaEntry | None,
                 params: dict,
                 context: dict) -> pd.Series | pd.DataFrame:
    imputer = KNNImputer(n_neighbors=params.get("n_neighbors", 5))
    features = params.get("features", None)
    target = params.get("target", None)
    if features is None or target is None:
        raise ValueError("KNN imputation requires 'features' and 'target' parameters.")
    if isinstance(data, pd.Series):
        raise ValueError("KNN imputation is not suitable for Series. Please provide a DataFrame with specified features and target.")
    else:
        df = data.copy()
    imputed_array = imputer.fit_transform(df[features + [target]])
    imputed_df = pd.DataFrame(imputed_array, columns=features + [target], index=df.index)
    return imputed_df[target]
    
TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_knn",
        fn=impute_knn_fn,
        allowed_params={"n_neighbors": int, "features": list, "target": str},
        description="Fill missing values using KNN imputation.",

        # Compatibility rules
        allowed_base=["numeric"],
        allowed_subtype=["continuous", "discrete"],
    )
)

def impute_constant_fn(data: pd.Series | pd.DataFrame,
                        schema: SchemaEntry | None,
                        params: dict,
                        context: dict) -> pd.Series | pd.DataFrame:
    constant_value = params.get("constant_value", 0)
    return data.fillna(constant_value)

TRANSFORM_REGISTRY.register(
    "missing",
    TransformationDefinition(
        name="impute_constant",
        fn=impute_constant_fn,
        allowed_params={"constant_value": (any)},
        description="Fill missing values with a constant value.",
    )
)


