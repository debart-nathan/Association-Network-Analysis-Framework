import numpy as np
from ...utils.metrics import (
    safe_entropy,
)

def build_numeric_metadata(s, info):
    n = s.count()
    if n == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "median": None}

    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
    }


def build_boolean_metadata(s, info):
    counts = s.value_counts(dropna=True).to_dict()
    return {
        "value_counts": {str(k): int(v) for k, v in counts.items()},
        "entropy": safe_entropy(s),
    }


def build_datetime_metadata(s, info):
    n = s.count()
    if n == 0:
        return {"min": None, "max": None, "range_days": None}

    min_dt = s.min()
    max_dt = s.max()

    return {
        "min": min_dt.isoformat() if hasattr(min_dt, "isoformat") else min_dt,
        "max": max_dt.isoformat() if hasattr(max_dt, "isoformat") else max_dt,
        "range_days": (max_dt - min_dt).days,
    }


def build_categorical_metadata(s, info):
    counts = s.value_counts(dropna=True).head(20).to_dict()
    return {
        "top_values": {str(k): int(v) for k, v in counts.items()},
        "entropy": safe_entropy(s),
    }


def build_text_metadata(s, info):
    lengths = s.astype(str).str.len()
    n = lengths.count()

    return {
        "avg_length": float(lengths.mean()) if n else None,
        "max_length": int(lengths.max()) if n else None,
        "entropy": safe_entropy(s),
    }


def build_structured_metadata(s, info):
    n = s.count()
    if n == 0:
        return {"avg_json_depth": None, "avg_keys": None}

    def json_depth(x):
        if isinstance(x, dict):
            return 1 + max((json_depth(v) for v in x.values()), default=0)
        elif isinstance(x, list):
            return 1 + max((json_depth(i) for i in x), default=0)
        else:
            return 0

    depths = s.dropna().apply(json_depth)
    keys = s.dropna().apply(lambda x: len(x) if isinstance(x, dict) else 0)

    return {
        "avg_json_depth": float(depths.mean()),
        "avg_keys": float(keys.mean()),
    }