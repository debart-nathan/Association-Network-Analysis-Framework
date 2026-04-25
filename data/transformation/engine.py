from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data.transformation.registry import TRANSFORM_REGISTRY, TransformationResult, TransformationDefinition
from data.transformation.plan import TransformationPlan, TransformStep
from data.transformation.validation import validate_plan
from data.transformation.dag import toposort_steps
from data.schema.normalized import NormalizedDataFrame
from data.schema.schema_types import SchemaEntry
from data.schema.inference import infer_type
from data.schema.metadata import build_metadata


@dataclass
class EngineContext:
    df: pd.DataFrame
    schema: dict[str, SchemaEntry]
    metadata: dict[str, dict]

    def merge(self, result: TransformationResult) -> "EngineContext":
        # Drop columns
        for col in result.drop_columns:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
            self.schema.pop(col, None)
            self.metadata.pop(col, None)

        # Add new columns (overwrite forbidden)
        for col_name, series in result.new_columns.items():
            if col_name in self.df.columns:
                raise RuntimeError(
                    f"Transformation attempted to create column '{col_name}' "
                    f"which already exists (overwrite is forbidden)."
                )

            self.df[col_name] = series

            # Schema: use provided or infer
            if col_name in result.new_schema:
                col_schema = result.new_schema[col_name]
            else:
                col_schema = infer_type(series)

            self.schema[col_name] = col_schema

            # Metadata: use provided or build
            if col_name in result.new_metadata:
                col_metadata = result.new_metadata[col_name]
            else:
                col_metadata = build_metadata(
                    self.df[[col_name]], {col_name: col_schema}
                )[col_name]

            self.metadata[col_name] = col_metadata

        # Apply any schema updates for existing columns
        for col_name, col_schema in result.new_schema.items():
            if col_name in self.df.columns and col_name not in result.new_columns:
                self.schema[col_name] = col_schema

        # Apply any metadata updates for existing columns
        for col_name, col_metadata in result.new_metadata.items():
            if col_name in self.df.columns and col_name not in result.new_columns:
                self.metadata[col_name] = col_metadata

        return self


def _apply_single_step(
    ctx: EngineContext,
    step: TransformStep,
) -> tuple[TransformationResult, TransformationDefinition]:
    """
    Internal helper to fetch the transformation definition, validate, and execute it.
    """
    definition = TRANSFORM_REGISTRY.get(step.category, step.name)
    definition.validate_params(step.params)

    # Ensure all input columns exist at this point in the pipeline
    missing = [c for c in step.inputs if c not in ctx.df.columns]
    if missing:
        raise RuntimeError(
            f"Transformation '{step.category}:{step.name}' requires missing columns: {missing}"
        )

    result = definition.fn(ctx, step.inputs, step.params)

    if not isinstance(result, TransformationResult):
        raise RuntimeError(
            f"Transformation '{step.category}:{step.name}' must return TransformationResult"
        )

    return result, definition


def apply_transformations(
    ndf: NormalizedDataFrame,
    plan: TransformationPlan
) -> NormalizedDataFrame:
    """
    Apply a full transformation plan to a NormalizedDataFrame.
    Returns a NEW NormalizedDataFrame with updated df, schema, and metadata.

    This engine:
    - Executes a global DAG of TransformStep (multi-column capable).
    - Forbids overwriting existing columns (new columns only).
    - Delegates shape and behavior to transformation definitions.
    """

    validate_plan(plan)

    # Immutable copies into engine context
    ctx = EngineContext(
        df=ndf.df.copy(),
        schema=dict(ndf.schema),
        metadata=dict(ndf.metadata),
    )

    # Global DAG ordering of all steps
    ordered_steps = toposort_steps(plan.steps)

    for step in ordered_steps:
        result, definition = _apply_single_step(ctx, step)
        ctx = ctx.merge(result)

        if result.terminal:
            break

    return NormalizedDataFrame(df=ctx.df, schema=ctx.schema, metadata=ctx.metadata)
