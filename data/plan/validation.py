from data.transformation.registry import TRANSFORM_REGISTRY
from data.filtering.registry import FILTER_REGISTRY
from data.plan.base_step import BaseStep
from data.plan.transform_step import TransformStep
from data.plan.filter_step import FilterStep
from data.plan.schema_cast_step import SchemaCastStep


# ============================================================
# Step-level validation
# ============================================================

def validate_step(step: BaseStep):
    """
    Validate a single step based on its step_type.

    Supported:
    - transform
    - filter
    - schema_cast
    """


    # Common validation
    if not isinstance(step.inputs, list):
        raise ValueError(f"Step '{step.id}' has invalid inputs (must be list).")

    # TransformStep validation
    if step.step_type == "transform":
        if not step.inputs:
            raise ValueError(
                f"TransformStep '{step.id}' must declare at least one input column."
            )

        definition = TRANSFORM_REGISTRY.get(step.category, step.name)
        if definition is None:
            raise ValueError(
                f"Unknown transformation '{step.category}:{step.name}' "
                f"in step '{step.id}'."
            )

        if definition.validate_params is not None:
            definition.validate_params(step.params)
        return


    # FilterStep validation
    if step.step_type == "filter":
        definition = FILTER_REGISTRY.get(step.category, step.name)
        if definition is None:
            raise ValueError(
                f"Unknown filter '{step.category}:{step.name}' "
                f"in step '{step.id}'."
            )

        if definition.validate_params is not None:
            definition.validate_params(step.params)
        return

    # SchemaCastStep validation
    if step.step_type == "schema_cast":
        # Must cast exactly one column
        if len(step.inputs) != 1:
            raise ValueError(
                f"SchemaCastStep '{step.id}' must have exactly one input column."
            )

        # Must have base + subtype
        if "base" not in step.params or "subtype" not in step.params:
            raise ValueError(
                f"SchemaCastStep '{step.id}' must define 'base' and 'subtype' params."
            )

        return

    # Unknown step type
    raise ValueError(f"Unknown step_type '{step.step_type}' in step '{step.id}'.")


# ============================================================
# Plan-level validation
# ============================================================

def validate_plan(plan):
    """
    Validate the full plan.

    Checks:
    - unique step IDs
    - each step is valid
    - all 'after' dependencies exist
    """

    # 1. Unique IDs
    ids = [s.id for s in plan.steps]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate step IDs found: {duplicates}")

    step_ids = set(ids)

    # 2. Validate each step
    for step in plan.steps:
        validate_step(step)

        # Validate dependencies
        if step.after:
            missing = [dep for dep in step.after if dep not in step_ids]
            if missing:
                raise ValueError(
                    f"Step '{step.id}' depends on missing steps: {missing}"
                )
