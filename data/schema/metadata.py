import pandas as pd
from .registry import registry


def build_metadata(df, schema):
    metadata = {}

    for col, entry in schema.items():
        base = entry.base
        subtype = entry.subtype

        # get the correct metadata builder from registry
        td = registry.get(base)
        builder = td.metadata_builder

        original = df[col]
        clean = original.dropna()

        metadata[col] = builder(
            original_series=original,
            clean_series=clean,
            subtype=subtype,
            confidence=entry.confidence,
        )

    return metadata
