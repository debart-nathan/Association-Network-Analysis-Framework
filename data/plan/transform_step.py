from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from data.plan.base_step import BaseStep


@dataclass(frozen=True)
class TransformStep(BaseStep):
    @staticmethod
    def create(id, category, name, inputs, params, label=None, after=None):
        return TransformStep(
            id=id,
            label=label,
            step_type="transform",
            category=category,
            name=name,
            inputs=inputs,
            params=params,
            after=after,
        )


