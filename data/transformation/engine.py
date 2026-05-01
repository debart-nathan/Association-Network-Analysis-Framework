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
from data.schema.registry import TYPE_REGISTRY, get_subtype


@dataclass
class EngineContext:
    df: pd.DataFrame
    schema: dict[str, SchemaEntry]
    metadata: dict[str, dict]

    def merge(
        self,
        result: TransformationResult,
        definition: Optional[TransformationDefinition] = None,
    ) -> "EngineContext":
        # Immutable copies
        new_df = self.df.copy()
        new_schema = dict(self.schema)
        new_metadata = dict(self.metadata)

        # Drop columns
        for col in result.drop_columns:
            if col not in new_df.columns:
                raise RuntimeError(
                    f"Transformation attempted to drop non-existent column '{col}'."
                )
            new_df = new_df.drop(columns=[col])
            new_schema.pop(col, None)
            new_metadata.pop(col, None)

        # Add new columns (overwrite forbidden)
        for col_name, series in result.new_columns.items():
            if col_name in new_df.columns:
                raise RuntimeError(
                    f"Transformation attempted to create column '{col_name}' "
                    f"which already exists (overwrite is forbidden)."
                )

            if len(series) != len(new_df):
                raise RuntimeError(
                    f"New column '{col_name}' has length {len(series)} "
                    f"but df has length {len(new_df)}."
                )

            # Schema: provided → output_schema → infer
            if col_name in result.new_schema:
                col_schema = result.new_schema[col_name]

            elif definition is not None and getattr(definition, "output_schema", None):
                base, subtype_name = definition.output_schema

                col_schema = SchemaEntry(
                    base=base,
                    subtype=subtype_name,   # keep string, consistent with infer_type
                    confidence=1.0,
                    candidates=[],
                    forced=True,
                )

            else:
                col_schema = infer_type(series)

            # Validate base exists
            TYPE_REGISTRY.get(col_schema.base)

            # Cast series according to schema (registry-driven)
            series = _apply_output_schema_casting(series, col_schema)

            new_df[col_name] = series
            new_schema[col_name] = col_schema

            # Metadata: provided or build
            if col_name in result.new_metadata:
                col_metadata = result.new_metadata[col_name]
            else:
                col_metadata = build_metadata(
                    new_df[[col_name]], {col_name: col_schema}
                )[col_name]

            if not isinstance(col_metadata, dict):
                raise RuntimeError(
                    f"Metadata for new column '{col_name}' must be a dict."
                )

            new_metadata[col_name] = col_metadata

        # Apply schema updates for existing columns
        for col_name, col_schema in result.new_schema.items():
            if col_name in result.new_columns:
                continue
            if col_name in new_df.columns:
                inferred = infer_type(new_df[col_name])

                if inferred.base != col_schema.base:
                    raise RuntimeError(
                        f"Schema update for '{col_name}' incompatible with actual dtype: "
                        f"{inferred.base} vs {col_schema.base}."
                    )

                TYPE_REGISTRY.get(col_schema.base)
                new_schema[col_name] = col_schema

        # Apply metadata updates for existing columns
        for col_name, col_metadata in result.new_metadata.items():
            if col_name in result.new_columns:
                continue
            if col_name in new_df.columns:
                if not isinstance(col_metadata, dict):
                    raise RuntimeError(
                        f"Metadata update for '{col_name}' must be a dict."
                    )
                new_metadata[col_name] = col_metadata

        return EngineContext(
            df=new_df,
            schema=new_schema,
            metadata=new_metadata,
        )


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
        ctx = ctx.merge(result, definition)

        if result.terminal:
            break

    return NormalizedDataFrame(df=ctx.df, schema=ctx.schema, metadata=ctx.metadata)


def _apply_output_schema_casting(series: pd.Series, schema: SchemaEntry) -> pd.Series:
    type_def = TYPE_REGISTRY.get(schema.base)

    # Convert subtype string → Subtype enum
    subtype_enum = get_subtype(schema.base, schema.subtype)

    if getattr(type_def, "cast", None) is None:
        return series

    if not callable(type_def.cast):
        raise RuntimeError(
            f"Type definition for base '{schema.base}' has non-callable 'cast' attribute."
        )
    return type_def.cast(series, subtype_enum)