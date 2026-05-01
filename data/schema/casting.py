import pandas as pd
from data.schema.subtypes import Subtype


def cast_numeric(series: pd.Series, subtype: Subtype) -> pd.Series:
    if subtype == Subtype.CONTINUOUS:
        return series.astype(float)

    if subtype == Subtype.DISCRETE:
        if pd.api.types.is_float_dtype(series):
            if (series.dropna() == series.dropna().astype(int)).all():
                return series.astype("Int64")
        return series

    return series


def cast_categorical(series: pd.Series, subtype: Subtype) -> pd.Series:
    # nominal / ordinal both map to string dtype
    return series.astype("string")


def cast_boolean(series: pd.Series, subtype: Subtype) -> pd.Series:
    return series.astype(bool)


def cast_datetime(series: pd.Series, subtype: Subtype) -> pd.Series:
    return pd.to_datetime(series)


def cast_text(series: pd.Series, subtype: Subtype) -> pd.Series:
    return series.astype("string")

def cast_identity(series: pd.Series, subtype: Subtype) -> pd.Series:
    return series
