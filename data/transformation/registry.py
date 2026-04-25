from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Any

import pandas as pd
from data.schema.schema_types import SchemaEntry


@dataclass(frozen=True)
class TransformationResult:
    new_columns: Dict[str, pd.Series] = field(default_factory=dict)
    drop_columns: List[str] = field(default_factory=list)
    new_schema: Dict[str, SchemaEntry] = field(default_factory=dict)
    new_metadata: Dict[str, dict] = field(default_factory=dict)
    terminal: bool = False


TransformationFn = Callable[
    [Any, List[str], Dict[str, Any]],
    TransformationResult,
]


@dataclass(frozen=True)
class TransformationDefinition:
    name: str
    fn: TransformationFn
    allowed_params: Dict[str, Any]
    description: str = ""
    is_derived: bool = False

    allowed_base: Optional[List[str]] = None
    allowed_subtype: Optional[List[str]] = None
    blocked_base: Optional[List[str]] = None
    blocked_subtype: Optional[List[str]] = None

    def validate_params(self, params: dict):
        unknown = set(params) - set(self.allowed_params)
        if unknown:
            raise ValueError(
                f"Unknown parameters for transformation '{self.name}': {unknown}. "
                f"Allowed: {list(self.allowed_params.keys())}"
            )

    def validate_schema(self, schema: SchemaEntry | None):
        if schema is None:
            return

        if self.blocked_base and schema.base in self.blocked_base:
            raise TypeError(
                f"Transformation '{self.name}' cannot be applied to base type '{schema.base}'."
            )

        if self.blocked_subtype and schema.subtype in self.blocked_subtype:
            raise TypeError(
                f"Transformation '{self.name}' cannot be applied to subtype '{schema.subtype}'."
            )

        if self.allowed_base and schema.base not in self.allowed_base:
            raise TypeError(
                f"Transformation '{self.name}' only applies to base types {self.allowed_base}, "
                f"but column is '{schema.base}'."
            )

        if self.allowed_subtype and schema.subtype not in self.allowed_subtype:
            raise TypeError(
                f"Transformation '{self.name}' only applies to subtypes {self.allowed_subtype}, "
                f"but column is '{schema.subtype}'."
            )


class TransformationRegistry:
    def __init__(self):
        self._registry: Dict[str, Dict[str, TransformationDefinition]] = {}

    def register(self, category: str, definition: TransformationDefinition):
        if not category:
            raise ValueError("Category name cannot be empty")

        self._registry.setdefault(category, {})

        if definition.name in self._registry[category]:
            raise ValueError(
                f"Transformation '{definition.name}' already registered "
                f"in category '{category}'"
            )

        self._registry[category][definition.name] = definition

    def get(self, category: str, name: str) -> TransformationDefinition:
        if category not in self._registry:
            raise KeyError(f"Unknown transformation category '{category}'")
        try:
            return self._registry[category][name]
        except KeyError:
            raise KeyError(f"Unknown transformation '{category}:{name}'")

    def get_or_none(self, category: str, name: str) -> Optional[TransformationDefinition]:
        return self._registry.get(category, {}).get(name)

    def categories(self) -> list[str]:
        return list(self._registry.keys())

    def list_transformations(self, category: str) -> list[str]:
        return list(self._registry.get(category, {}).keys())

    def all(self) -> Dict[str, Dict[str, TransformationDefinition]]:
        return self._registry

    def __contains__(self, category: str) -> bool:
        return category in self._registry

    def __repr__(self):
        return f"TransformationRegistry(categories={list(self._registry.keys())})"


TRANSFORM_REGISTRY = TransformationRegistry()
