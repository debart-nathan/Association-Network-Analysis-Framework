from typing import Any, Dict
import pandas as pd

from data.schema.schema_types import SchemaEntry, SchemaCandidate
from data.schema.registry import TYPE_REGISTRY
from data.schema.subtypes import Subtype


def infer_type(series: pd.Series, **kwargs) -> SchemaEntry:


    original = series
    s = series.dropna()

    candidates = []

    # run inference for each registered type
    for name, td in TYPE_REGISTRY.all():
        result = td.infer(s, **kwargs)
        if result is None:
            continue

        subtype_key, score = result

        # per-type minimum score (default 0.2)
        min_score = getattr(td, "min_score", 0.2)
        if score < min_score:
            continue

        # optional validation hook
        if td.validate and not td.validate(original):
            continue

        candidates.append(
            {
                "score": float(score),
                "priority": td.priority,
                "type_def": td,
                "subtype_key": subtype_key,
            }
        )

    # choose best candidate
    if candidates:
        best = max(candidates, key=lambda c: (c["score"], c["priority"]))
        td = best["type_def"]
        subtype_key = best["subtype_key"]

        return SchemaEntry(
            base=td.base,
            subtype=td.subtypes[subtype_key].name,
            confidence=best["score"],
            candidates=[
                SchemaCandidate(
                    base=c["type_def"].base,
                    subtype=c["subtype_key"].name,
                    score=c["score"],
                    priority=c["priority"],
                )
                for c in candidates
            ],
        )

    # fallback to unknown
    td = TYPE_REGISTRY.get("unknown")
    return SchemaEntry(
        base=td.base,
        subtype=td.subtypes[Subtype.NONE].name,
        confidence=0.0,
        candidates=[],
    )

