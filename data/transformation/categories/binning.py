from data.transformation.registry import TRANSFORM_REGISTRY, TransformationDefinition, TransformationResult
from data.schema.schema_types import SchemaEntry

import pandas as pd
import numpy as np
from typing import Literal, Sequence, Union


# ============================================================
# Helpers
# ============================================================

def _validate_bins(bins):
    if bins is None or bins < 2:
        raise ValueError("'bins' must be an integer >= 2")


def _scale_bins(extracted, bins, out_min, out_max):
    return extracted / (bins - 1) * (out_max - out_min) + out_min


def _finalize(col, extracted, ctx):
    col_name = f"{col}_binned"
    new_col = pd.Series(extracted, index=ctx.df.index, name=col_name)
    return TransformationResult(new_columns={col_name: new_col}, terminal=True)


Label = str


# ============================================================
# SAFE CUT (VERSION FINALE)
# ============================================================

def _safe_cut(s, bins, *,right=True, include_lowest=True):
    clean_bins = []

    for b in bins:
        try:
            bf = float(b)
            if not np.isnan(bf):
                clean_bins.append(bf)
        except:
            continue

    clean_bins = sorted(set(clean_bins))

    if len(clean_bins) < 2:
        return np.full(len(s), -1, dtype=int)

    return pd.cut(
        s,
        bins=clean_bins,
        right=right,
        retbins=False,
        include_lowest=include_lowest,
    ).to_numpy(dtype=int)





# ============================================================
# EQUAL WIDTH BINNING
# ============================================================

def equal_width_binning_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    bins = params.get("bins", 5)
    out_min = params.get("out_min", 0.0)
    out_max = params.get("out_max", 1.0)

    if bins < 2 or s.nunique(dropna=True) <= 1:
        extracted = np.full(len(s), out_min, dtype=float)
    else:
        data_min, data_max = s.min(), s.max()
        edges = np.linspace(data_min, data_max, bins + 1).tolist()
        extracted = _safe_cut(s, edges)
        extracted = extracted / (bins - 1) * (out_max - out_min) + out_min

    col_name = f"{col}_binned"
    return TransformationResult(
        new_columns={col_name: pd.Series(extracted, index=s.index, name=col_name)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "numeric",
    TransformationDefinition(
        name="equal_width_binning",
        fn=equal_width_binning_fn,
        allowed_params={"bins": int, "out_min": float, "out_max": float},
        description="Perform equal width binning on a numeric column.",
        is_derived=True,
        allowed_base=["numeric"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# QUANTILE BINNING
# ============================================================

def quantile_binning_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    bins = params.get("bins", 5)
    out_min = params.get("out_min", 0.0)
    out_max = params.get("out_max", 1.0)

    if bins < 2:
        extracted = np.full(len(s), out_min, dtype=float)
    else:
        q = s.quantile(np.linspace(0, 1, bins + 1)).values
        q = sorted(set(float(v) for v in q if not np.isnan(v)))

        if len(q) < 2:
            extracted = np.full(len(s), out_min, dtype=float)
        else:
            effective_bins = len(q) - 1
            extracted = _safe_cut(s, q)
            extracted = extracted / (effective_bins - 1) * (out_max - out_min) + out_min

    col_name = f"{col}_binned"
    return TransformationResult(
        new_columns={col_name: pd.Series(extracted, index=s.index, name=col_name)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "numeric",
    TransformationDefinition(
        name="quantile_binning",
        fn=quantile_binning_fn,
        allowed_params={"bins": int, "out_min": float, "out_max": float},
        description="Perform quantile binning on a numeric column.",
        is_derived=True,
        allowed_base=["numeric"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# KMEANS BINNING
# ============================================================

def kmeans_binning_fn(ctx, inputs, params):
    from sklearn.cluster import KMeans

    col = inputs[0]
    s = ctx.df[col].values.reshape(-1, 1)

    bins = params.get("bins", 5)
    out_min = params.get("out_min", 0.0)
    out_max = params.get("out_max", 1.0)

    if bins < 2:
        extracted = np.full(len(s), out_min, dtype=float)
    else:
        kmeans = KMeans(n_clusters=bins, random_state=42, n_init="auto")
        kmeans.fit(s)

        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        ordered_labels = np.searchsorted(order, kmeans.labels_)

        extracted = ordered_labels / (bins - 1) * (out_max - out_min) + out_min

    col_name = f"{col}_binned"
    return TransformationResult(
        new_columns={col_name: pd.Series(extracted, index=ctx.df.index, name=col_name)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "numeric",
    TransformationDefinition(
        name="kmeans_binning",
        fn=kmeans_binning_fn,
        allowed_params={"bins": int, "out_min": float, "out_max": float},
        description="Perform k-means binning on a numeric column.",
        is_derived=True,
        allowed_base=["numeric"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# DOMAIN SPECIFIC BINNING
# ============================================================

def domain_specific_binning_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col]

    bins = params.get("bins")
    out_min = params.get("out_min", 0.0)
    out_max = params.get("out_max", 1.0)

    if not bins or len(bins) < 2:
        raise ValueError("Domain specific binning requires at least 2 bin edges.")

    bins = sorted(float(b) for b in bins if not np.isnan(float(b)))
    effective_bins = len(bins) - 1

    extracted = _safe_cut(s, bins)
    extracted = extracted / (effective_bins - 1) * (out_max - out_min) + out_min

    col_name = f"{col}_binned"
    return TransformationResult(
        new_columns={col_name: pd.Series(extracted, index=s.index, name=col_name)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "numeric",
    TransformationDefinition(
        name="domain_specific_binning",
        fn=domain_specific_binning_fn,
        allowed_params={"bins": list, "out_min": float, "out_max": float},
        description="Perform domain specific binning on a numeric column.",
        is_derived=True,
        allowed_base=["numeric"],
        output_schema=("numeric", "discrete"),
    )
)
