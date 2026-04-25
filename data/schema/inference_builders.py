import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from data.schema.subtypes import Subtype

from utils.metrics import (
    cardinality_ratio,
    avg_length,
    safe_entropy,
    normalized_entropy,
    is_monotonic,
)

ScoreResult = Optional[Tuple[Subtype, float]]

# -------------------------------------------------------------------
# BOOLEAN
# -------------------------------------------------------------------

BOOLEAN_TRUE = {"true", "t", "yes", "y", "1"}
BOOLEAN_FALSE = {"false", "f", "no", "n", "0"}

def infer_boolean(s: pd.Series) -> ScoreResult:
    s_clean = s.dropna()

    # dtype bool → perfect match
    if pd.api.types.is_bool_dtype(s_clean):
        return Subtype.NONE, 1.0

    # integer booleans
    if pd.api.types.is_integer_dtype(s_clean):
        unique_vals = set(s_clean.unique())
        if unique_vals <= {0, 1}:
            return Subtype.NONE, 0.95

    # string booleans
    if pd.api.types.is_string_dtype(s_clean) or s_clean.dtype == object:
        lowered = s_clean.astype(str).str.lower()
        valid = lowered.isin(BOOLEAN_TRUE | BOOLEAN_FALSE)
        purity = valid.mean()

        if purity > 0.95:
            return Subtype.NONE, float(0.8 + 0.2 * purity)

    return None


# -------------------------------------------------------------------
# DATETIME
# -------------------------------------------------------------------

def infer_datetime(s: pd.Series) -> ScoreResult:
    s_clean = s.dropna()

    # Already datetime dtype
    if pd.api.types.is_datetime64_any_dtype(s_clean):
        return Subtype.NONE, 0.95

    # Only strings/objects can be parsed
    if not (pd.api.types.is_string_dtype(s_clean) or s_clean.dtype == object):
        return None

    sample = s_clean.sample(min(20, len(s_clean)), random_state=0).astype(str)

    successes = 0
    for val in sample:
        try:
            pd.to_datetime(val, errors="raise")
            successes += 1
        except Exception:
            pass

    ratio = successes / len(sample)

    if ratio >= 0.8:
        return Subtype.NONE, float(0.7 + 0.3 * ratio)

    return None


# -------------------------------------------------------------------
# NUMERIC
# -------------------------------------------------------------------

def infer_numeric(s: pd.Series) -> ScoreResult:
    s_clean = s.dropna()

    # Exclude booleans
    if pd.api.types.is_bool_dtype(s_clean):
        return None

    if not pd.api.types.is_numeric_dtype(s_clean):
        return None

    ratio = cardinality_ratio(s_clean)

    # Very low cardinality → probably categorical
    if ratio < 0.02:
        return Subtype.DISCRETE, 0.2

    # Integer discrete
    if pd.api.types.is_integer_dtype(s_clean) and ratio < 0.5:
        score = 0.6 + 0.3 * ratio
        return Subtype.DISCRETE, float(score)

    # Continuous numeric
    score = 0.7 + 0.2 * (1 - ratio)
    return Subtype.CONTINUOUS, float(score)


# -------------------------------------------------------------------
# CATEGORICAL
# -------------------------------------------------------------------

def infer_categorical(s: pd.Series) -> ScoreResult:
    s_clean = s.dropna().astype(str)

    ratio = cardinality_ratio(s_clean)
    ent = normalized_entropy(s_clean)

    # Too high cardinality → not categorical
    if ratio > 0.7:
        return None

    # Token analysis
    tokens = s_clean.str.split()
    avg_tokens = tokens.map(len).mean()

    # Character diversity
    char_div = s_clean.apply(lambda x: len(set(x))).mean()

    # Categorical scoring
    score = (
        0.5 * (1 - ratio) +
        0.3 * (1 - abs(ent - 0.6)) +
        0.2 * (1 / (1 + avg_tokens))
    )

    score = float(max(0.0, min(1.0, score)))

    if score < 0.2:
        return None

    if is_monotonic(s_clean):
        return Subtype.ORDINAL, min(1.0, score + 0.1)

    return Subtype.NOMINAL, score


# -------------------------------------------------------------------
# TEXT
# -------------------------------------------------------------------

def infer_text(s: pd.Series, *, text_length_threshold: int = 50) -> ScoreResult:
    s_clean = s.dropna().astype(str)

    ratio = cardinality_ratio(s_clean)
    avg_len = avg_length(s_clean)
    ent = normalized_entropy(s_clean)

    # Token count
    tokens = s_clean.str.split()
    avg_tokens = tokens.map(len).mean()

    # If low cardinality → categorical
    if ratio < 0.1:
        return None

    # Text scoring
    length_factor = min(1.0, avg_len / text_length_threshold)
    token_factor = min(1.0, avg_tokens / 5)

    score = (
        0.3 * length_factor +
        0.3 * token_factor +
        0.2 * ent +
        0.2 * ratio
    )

    score = float(max(0.0, min(1.0, score)))

    if avg_len > 200:
        return Subtype.LONG_TEXT, score

    return Subtype.SHORT_TEXT, score


# -------------------------------------------------------------------
# STRUCTURED (JSON)
# -------------------------------------------------------------------

def infer_structured(s: pd.Series) -> ScoreResult:
    s_clean = s.dropna().astype(str)

    if len(s_clean) == 0:
        return None

    sample = s_clean.sample(min(10, len(s_clean)), random_state=0)

    parsed_types = []
    for val in sample:
        try:
            parsed = json.loads(val)
            parsed_types.append(type(parsed))
        except Exception:
            pass

    if len(parsed_types) == 0:
        return None

    success_ratio = len(parsed_types) / len(sample)

    if success_ratio < 0.7:
        return None

    if all(t is dict for t in parsed_types):
        return Subtype.OBJECT, float(success_ratio)

    if all(t is list for t in parsed_types):
        return Subtype.ARRAY, float(success_ratio)

    return None
