
import numpy as np
import pandas as pd


def cardinality_ratio(s: pd.Series) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    return s.nunique(dropna=True) / n


def avg_length(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return float(s.astype(str).str.len().mean())


def safe_entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True, dropna=True)
    if len(counts) == 0:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


def normalized_entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True, dropna=True)
    if len(counts) <= 1:
        return 0.0
    ent = safe_entropy(series)
    max_ent = np.log2(len(counts))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def is_monotonic(series: pd.Series) -> bool:
    try:
        s = series.dropna()
        if len(s) <= 1:
            return False
        return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing)
    except Exception:
        return False
