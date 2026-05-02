from dataclasses import dataclass
from typing import Dict
from data.schema.schema_types import SchemaEntry

@dataclass(frozen=True)
class SchemaCastResult:
    new_schema: Dict[str, SchemaEntry]
    new_metadata: Dict[str, dict]
