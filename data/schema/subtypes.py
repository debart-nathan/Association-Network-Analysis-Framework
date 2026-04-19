from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Subtype(Enum):
    NONE = auto()
    CONTINUOUS = auto()
    DISCRETE = auto()
    SHORT_TEXT = auto()
    LONG_TEXT = auto()
    NOMINAL = auto()
    ORDINAL = auto()
    OBJECT = auto()
    ARRAY = auto()


@dataclass(frozen=True)
class SubtypeDefinition:
    name: str
    description: Optional[str] = None
