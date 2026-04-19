from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class SchemaCandidate:
    base: str
    subtype: str
    score: float
    priority: int

@dataclass(frozen=True)
class SchemaEntry:
    base: str
    subtype: str
    confidence: float
    candidates: List[SchemaCandidate]
    forced: bool = False # indicates if this schema was forced by the user
