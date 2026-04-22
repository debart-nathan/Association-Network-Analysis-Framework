import pandas as pd
from typing import Any

from .registry import TRANSFORM_REGISTRY
from .plan import TransformationPlan, ColumnTransformStep, DerivedSpec
from .validation import validate_plan
from .dag import toposort_steps
from ..schema.normalized import NormalizedDataFrame
from ..schema.schema_types import SchemaEntry
from ..schema.inference import infer_type


def _apply_single_step(
    series: pd.Series | pd.DataFrame,
    schema: SchemaEntry | None,
    metadata: dict,
    step: ColumnTransformStep,
) -> pd.Series | pd.DataFrame:
    definition = TRANSFORM_REGISTRY.get(step.category, step.name)
    definition.validate_params(step.params)
    return definition.fn(series, schema, metadata, step.params)


def apply_transformations(
    ndf: NormalizedDataFrame,
    plan: TransformationPlan
) -> pd.DataFrame:
    """
    Apply a full transformation plan to a NormalizedDataFrame.
    """
    # Validate plan against registry before execution
    validate_plan(plan)

    df = ndf.df.copy()

    for col, spec in plan.columns.items():
        if col not in df.columns:
            continue

        schema: SchemaEntry = ndf.schema[col]
        metadata: dict = ndf.metadata[col]
        series: pd.Series | pd.DataFrame = df[col]

        # DAG-based ordering of steps
        ordered_steps = toposort_steps(spec.steps)

        for step in ordered_steps:
            # Shape-changing categories handled specially
            if step.category == "encoding":
                encoded = _apply_single_step(series, schema, metadata, step)
                df = df.drop(columns=[col]).join(encoded)
                break  

            elif step.category == "datetime":
                # datetime decomposition may add multiple columns
                new_col = _apply_single_step(series, schema, metadata, step)
                df[f"{col}_{step.name}"] = new_col

            else:
                # regular single-column transform
                series = _apply_single_step(series, schema, metadata, step)

        # If encoding or datetime dropped the column, ensure it's removed
        if "encoding" in [s.category for s in ordered_steps]:
            # already dropped and replaced
            continue

        if "datetime" in [s.category for s in ordered_steps]:
            # drop original datetime column after decomposition
            df = df.drop(columns=[col])
            continue

        # Assign transformed series back
        df[col] = series

    # Derived transformations (multi-column, expression-based)
    if plan.derived:
        for d in plan.derived:
            definition = TRANSFORM_REGISTRY.get("derived", d.name)
            if not definition.is_derived:
                raise ValueError(f"Transformation '{d.name}' is not marked as derived")

            definition.validate_params(d.params)

            result = definition.fn(df, None, {}, d.params)

            if isinstance(result, pd.Series):
                df[d.new_col] = result
            else:
                df = df.join(result)

    return df
