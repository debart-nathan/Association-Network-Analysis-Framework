from data.transformation.registry import TRANSFORM_REGISTRY
from data.transformation.plan import TransformationPlan, TransformStep


def validate_step(step: TransformStep):
    """
    Validate a single TransformStep against the registry.

    Checks:
    - category/name exists in registry
    - params are valid for that transform
    - inputs list is non-empty
    """
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

    definition.validate_params(step.params)


def validate_plan(plan: TransformationPlan):
    """
    Validate the full transformation plan.

    Checks:
    - every step is valid
    - no duplicate step IDs
    - all referenced 'after' dependencies exist
    - registry categories/names are valid
    """

    # 1. Validate unique step IDs
    ids = [s.id for s in plan.steps]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate step IDs found: {duplicates}")

    # Build lookup for dependency validation
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
