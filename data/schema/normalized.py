from dataclasses import dataclass
from typing import Dict
import pandas as pd

from data.schema.inference import infer_type
from data.schema.metadata import build_metadata
from data.schema.schema_types import SchemaEntry  #

@dataclass(frozen=True)
class NormalizedDataFrame:
    df: pd.DataFrame
    schema: Dict[str, SchemaEntry]
    metadata: dict


def build_NormalizedDataFrame(df: pd.DataFrame) -> NormalizedDataFrame:
    schema = {col: infer_type(df[col]) for col in df.columns}
    metadata = build_metadata(df, schema)
    return NormalizedDataFrame(df=df, schema=schema, metadata=metadata)
