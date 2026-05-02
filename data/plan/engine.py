from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data.plan.base_step import BaseStep
from data.plan.dag import toposort_steps
from data.plan.validation import validate_plan

from data.schema.normalized import NormalizedDataFrame
from data.schema.schema_types import SchemaEntry
from data.schema.inference import infer_type
from data.schema.metadata import build_metadata
from data.schema.registry import TYPE_REGISTRY, get_subtype

from data.transformation.registry import (
    TRANSFORM_REGISTRY,
    TransformationResult,
    TransformationDefinition,
)

from data.filtering.registry import (
    FILTER_REGISTRY,
    FilterResult,
    FilterDefinition,
)

from data.schema.cast_result import SchemaCastResult


# ============================================================
# Engine Context
# ============================================================

@dataclass
class EngineContext:
    df: pd.DataFrame
    schema: dict[str, SchemaEntry]
    metadata: dict[str, dict]

    def merge(self, result, definition=None) -> "EngineContext":

        new_df = self.df.copy()
        new_schema = dict(self.schema)
        new_metadata = dict(self.metadata)

        # -------------------------
        # 1. Row filtering
        # -------------------------
        if getattr(result, "drop_rows", None) is not None:
            mask = result.drop_rows
            if len(mask) != len(new_df):
                raise RuntimeError("Row mask length mismatch.")
            new_df = new_df[mask].reset_index(drop=True)

        # -------------------------
        # 2. Column dropping
        # -------------------------
        for col in getattr(result, "drop_columns", []) or []:
            if col not in new_df.columns:
                raise RuntimeError(f"Attempted to drop non-existent column '{col}'.")
            new_df = new_df.drop(columns=[col])
            new_schema.pop(col, None)
            new_metadata.pop(col, None)

        # -------------------------
        # 3. New columns (TransformStep only)
        # -------------------------
        new_columns = getattr(result, "new_columns", None) or {}
        new_schema_updates = getattr(result, "new_schema", None) or {}
        new_metadata_updates = getattr(result, "new_metadata", None) or {}

        for col_name, series in new_columns.items():
            if col_name in new_df.columns:
                raise RuntimeError(f"Column '{col_name}' already exists.")

            if len(series) != len(new_df):
                raise RuntimeError(
                    f"New column '{col_name}' has length {len(series)} "
                    f"but df has length {len(new_df)}."
                )

            # Schema resolution
            if col_name in new_schema_updates:
                col_schema = new_schema_updates[col_name]

            elif definition is not None and getattr(definition, "output_schema", None):
                base, subtype_name = definition.output_schema
                col_schema = SchemaEntry(
                    base=base,
                    subtype=subtype_name,
                    confidence=1.0,
                    candidates=[],
                    forced=True,
                )

            else:
                col_schema = infer_type(series)

            TYPE_REGISTRY.get(col_schema.base)

            series = _apply_output_schema_casting(series, col_schema)

            new_df[col_name] = series
            new_schema[col_name] = col_schema

            # Metadata
            if col_name in new_metadata_updates:
                col_metadata = new_metadata_updates[col_name]
            else:
                col_metadata = build_metadata(
                    new_df[[col_name]], {col_name: col_schema}
                )[col_name]

            new_metadata[col_name] = col_metadata

        # -------------------------
        # 4. Schema updates (SchemaCastStep or TransformStep)
        # -------------------------
        for col_name, col_schema in new_schema_updates.items():
            if col_name in new_columns:
                continue
            if col_name in new_df.columns:
                inferred = infer_type(new_df[col_name])
                if inferred.base != col_schema.base:
                    raise RuntimeError(
                        f"Schema update for '{col_name}' incompatible: "
                        f"{inferred.base} vs {col_schema.base}."
                    )
                TYPE_REGISTRY.get(col_schema.base)
                new_schema[col_name] = col_schema

        # -------------------------
        # 5. Metadata updates
        # -------------------------
        for col_name, col_metadata in new_metadata_updates.items():
            if col_name in new_columns:
                continue
            if col_name in new_df.columns:
                new_metadata[col_name] = col_metadata

        return EngineContext(
            df=new_df,
            schema=new_schema,
            metadata=new_metadata,
        )


# ============================================================
# Step dispatchers
# ============================================================

def _apply_transform_step(ctx: EngineContext, step: BaseStep):
    definition: TransformationDefinition = TRANSFORM_REGISTRY.get(step.category, step.name)

    if definition.validate_params is not None:
        definition.validate_params(step.params)

    missing = [c for c in step.inputs if c not in ctx.df.columns]
    if missing:
        raise RuntimeError(
            f"Transform '{step.category}:{step.name}' missing columns: {missing}"
        )

    result = definition.fn(ctx, step.inputs, step.params)

    if not isinstance(result, TransformationResult):
        raise RuntimeError("Transform must return TransformationResult")

    return result, definition



def _apply_filter_step(ctx: EngineContext, step: BaseStep):
    definition: FilterDefinition = FILTER_REGISTRY.get(step.category, step.name)

    if definition.validate_params is not None:
        definition.validate_params(step.params)

    result = definition.fn(ctx, step.inputs, step.params)

    if not isinstance(result, FilterResult):
        raise RuntimeError("Filter must return FilterResult")

    return result, definition



def _apply_schema_cast_step(ctx: EngineContext, step: BaseStep):
    """
    No casting registry: use TYPE_REGISTRY directly.
    """
    if not step.inputs:
        raise RuntimeError(f"Schema cast step '{step.id}' has no input column.")

    col = step.inputs[0]
    if col not in ctx.df.columns:
        raise RuntimeError(f"Schema cast: missing column '{col}'")

    base = step.params["base"]
    subtype = step.params["subtype"]

    type_def = TYPE_REGISTRY.get(base)
    subtype_enum = get_subtype(base, subtype)

    if type_def.cast is None:
        raise RuntimeError(f"Type '{base}' does not support casting.")

    series = ctx.df[col]
    casted = type_def.cast(series, subtype_enum)

    new_schema = {
        col: SchemaEntry(
            base=base,
            subtype=subtype,
            confidence=1.0,
            candidates=[],
            forced=True,
        )
    }

    new_metadata = {
        col: build_metadata(
            pd.DataFrame({col: casted}),
            new_schema
        )[col]
    }

    return SchemaCastResult(
        new_schema=new_schema,
        new_metadata=new_metadata,
    ), None


# ============================================================
# Main engine
# ============================================================

def apply_plan(ndf: NormalizedDataFrame, plan) -> NormalizedDataFrame:

    validate_plan(plan)

    ctx = EngineContext(
        df=ndf.df.copy(),
        schema=dict(ndf.schema),
        metadata=dict(ndf.metadata),
    )

    ordered_steps = toposort_steps(plan.steps)

    for step in ordered_steps:

        if step.step_type == "transform":
            result, definition = _apply_transform_step(ctx, step)

        elif step.step_type == "filter":
            result, definition = _apply_filter_step(ctx, step)

        elif step.step_type == "schema_cast":
            result, definition = _apply_schema_cast_step(ctx, step)

        else:
            raise RuntimeError(f"Unknown step_type '{step.step_type}'")

        ctx = ctx.merge(result, definition)

        if getattr(result, "terminal", False):
            break

    return NormalizedDataFrame(
        df=ctx.df,
        schema=ctx.schema,
        metadata=ctx.metadata,
    )


# ============================================================
# Schema casting helper
# ============================================================

def _apply_output_schema_casting(series: pd.Series, schema: SchemaEntry) -> pd.Series:
    type_def = TYPE_REGISTRY.get(schema.base)
    subtype_enum = get_subtype(schema.base, schema.subtype)

    if type_def.cast is None:
        return series

    return type_def.cast(series, subtype_enum)
