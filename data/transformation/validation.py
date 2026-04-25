from data.transformation.registry import TRANSFORM_REGISTRY
from data.transformation.plan import TransformationPlan, ColumnTransformSpec, ColumnTransformStep, DerivedSpec


def validate_column_spec(spec: ColumnTransformSpec):
    for step in spec.steps:
        definition = TRANSFORM_REGISTRY.get(step.category, step.name)
        definition.validate_params(step.params)


def validate_plan(plan: TransformationPlan):
    # Validate column-level specs
    for col, spec in plan.columns.items():
        validate_column_spec(spec)

    # Validate derived specs
    if plan.derived:
        for d in plan.derived:
            definition = TRANSFORM_REGISTRY.get("derived", d.name)
            if not definition.is_derived:
                raise ValueError(
                    f"Transformation '{d.name}' in derived plan is not marked as derived"
                )
            definition.validate_params(d.params)
