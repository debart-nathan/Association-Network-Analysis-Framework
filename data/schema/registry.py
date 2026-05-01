from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple
import pandas as pd

from data.schema.subtypes import Subtype, SubtypeDefinition
from data.schema.inference_builders import (
    infer_boolean,
    infer_datetime,
    infer_numeric,
    infer_categorical,
    infer_text,
    infer_structured,
)
from data.schema.metadata_builders import (
    build_boolean_metadata,
    build_datetime_metadata,
    build_numeric_metadata,
    build_text_metadata,
    build_categorical_metadata,
    build_structured_metadata,
)

from data.schema.casting import (
    cast_identity,
    cast_numeric,
    cast_categorical,
    cast_boolean,
    cast_datetime,
    cast_text,
)


ScoreResult = Optional[Tuple[Subtype, float]]


@dataclass(frozen=True)
class TypeDefinition:
    base: str
    subtypes: Dict[Subtype, SubtypeDefinition]
    infer: Callable[..., ScoreResult]
    metadata_builder: Callable[..., Dict[str, Any]]
    validate: Optional[Callable[..., bool]] = None
    priority: int = 100
    cast: Optional[Callable[[pd.Series, Subtype], pd.Series]] = None


class TypeRegistry:
    def __init__(self):
        self._types: Dict[str, TypeDefinition] = {}

    def register(self, name: str, definition: TypeDefinition):
        if name in self._types:
            raise ValueError(f"Type '{name}' already registered")
        self._types[name] = definition

    def get(self, name: str) -> TypeDefinition:
        try:
            return self._types[name]
        except KeyError:
            raise ValueError(f"Unknown type '{name}'")

    def all(self):
        return self._types.items()


def get_subtype(base: str, subtype_name: str) -> Subtype:
    """
    Convert a subtype string into the correct Subtype enum,
    using TYPE_REGISTRY as the single source of truth.
    """
    type_def = TYPE_REGISTRY.get(base)

    for subtype_enum, subtype_def in type_def.subtypes.items():
        if subtype_def.name == subtype_name:
            return subtype_enum

    raise ValueError(
        f"Unknown subtype '{subtype_name}' for base '{base}'. "
        f"Valid subtypes: {[d.name for d in type_def.subtypes.values()]}"
    )


# -------------------------
# Registry Initialization
# -------------------------

TYPE_REGISTRY = TypeRegistry()

TYPE_REGISTRY.register(
    "boolean",
    TypeDefinition(
        base="boolean",
        subtypes={Subtype.NONE: SubtypeDefinition("boolean")},
        infer=infer_boolean,
        metadata_builder=build_boolean_metadata,
        priority=100,
        cast=cast_boolean,
    ),
)

TYPE_REGISTRY.register(
    "datetime",
    TypeDefinition(
        base="datetime",
        subtypes={Subtype.NONE: SubtypeDefinition("datetime")},
        infer=infer_datetime,
        metadata_builder=build_datetime_metadata,
        priority=90,
        cast=cast_datetime,
    ),
)

TYPE_REGISTRY.register(
    "structured",
    TypeDefinition(
        base="structured",
        subtypes={
            Subtype.OBJECT: SubtypeDefinition("object"),
            Subtype.ARRAY: SubtypeDefinition("array"),
        },
        infer=infer_structured,
        metadata_builder=build_structured_metadata,
        priority=80,
        cast=None,  # structured types are not cast
    ),
)

TYPE_REGISTRY.register(
    "numeric",
    TypeDefinition(
        base="numeric",
        subtypes={
            Subtype.CONTINUOUS: SubtypeDefinition("continuous"),
            Subtype.DISCRETE: SubtypeDefinition("discrete"),
        },
        infer=infer_numeric,
        metadata_builder=build_numeric_metadata,
        priority=70,
        cast=cast_numeric,
    ),
)

TYPE_REGISTRY.register(
    "categorical",
    TypeDefinition(
        base="categorical",
        subtypes={
            Subtype.NOMINAL: SubtypeDefinition("nominal"),
            Subtype.ORDINAL: SubtypeDefinition("ordinal"),
        },
        infer=infer_categorical,
        metadata_builder=build_categorical_metadata,
        priority=60,
        cast=cast_categorical,
    ),
)

TYPE_REGISTRY.register(
    "text",
    TypeDefinition(
        base="text",
        subtypes={
            Subtype.SHORT_TEXT: SubtypeDefinition("short_text"),
            Subtype.LONG_TEXT: SubtypeDefinition("long_text"),
        },
        infer=infer_text,
        metadata_builder=build_text_metadata,
        priority=50,
        cast=cast_text,
    ),
)

TYPE_REGISTRY.register(
    "unknown",
    TypeDefinition(
        base="unknown",
        subtypes={Subtype.NONE: SubtypeDefinition("unknown")},
        infer=lambda s, **_: None,
        metadata_builder=lambda **_: {},
        priority=0,
        cast=cast_identity,
    ),
)
