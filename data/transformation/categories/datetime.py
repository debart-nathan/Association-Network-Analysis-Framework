from data.transformation.registry import TRANSFORM_REGISTRY, TransformationDefinition, TransformationResult
import pandas as pd

# ============================================================
# Helper
# ============================================================

def _finalize(ctx, col, suffix, extracted):
    col_name = f"{col}_{suffix}"
    new_col = pd.Series(extracted, index=ctx.df.index, name=col_name)
    return TransformationResult(new_columns={col_name: new_col}, terminal=True)


# ============================================================
# EXTRACT YEAR
# ============================================================

def extract_year_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.year
    return _finalize(ctx, col, "year", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_year",
        fn=extract_year_fn,
        allowed_params={},
        description="Extract the year from a datetime column.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT MONTH
# ============================================================

def extract_month_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.month
    return _finalize(ctx, col, "month", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_month",
        fn=extract_month_fn,
        allowed_params={},
        description="Extract the month from a datetime column.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT DAY
# ============================================================

def extract_day_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.day
    return _finalize(ctx, col, "day", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_day",
        fn=extract_day_fn,
        allowed_params={},
        description="Extract the day from a datetime column.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT HOUR
# ============================================================

def extract_hour_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.hour
    return _finalize(ctx, col, "hour", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_hour",
        fn=extract_hour_fn,
        allowed_params={},
        description="Extract the hour from a datetime column.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT WEEKDAY
# ============================================================

def extract_weekday_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.weekday
    return _finalize(ctx, col, "weekday", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_weekday",
        fn=extract_weekday_fn,
        allowed_params={},
        description="Extract the weekday (0=Mon..6=Sun).",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT WEEKEND
# ============================================================

def extract_weekend_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = (pd.to_datetime(ctx.df[col], errors="coerce").dt.weekday >= 5).astype(int)
    return _finalize(ctx, col, "is_weekend", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_weekend",
        fn=extract_weekend_fn,
        allowed_params={},
        description="Return 1 if weekend, else 0.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT SEASON
# ============================================================

def extract_season_fn(ctx, inputs, params):
    col = inputs[0]
    month = pd.to_datetime(ctx.df[col], errors="coerce").dt.month
    extracted = month % 12 // 3 + 1
    return _finalize(ctx, col, "season", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_season",
        fn=extract_season_fn,
        allowed_params={},
        description="Extract season (1=Winter..4=Fall).",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT QUARTER
# ============================================================

def extract_quarter_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.quarter
    return _finalize(ctx, col, "quarter", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_quarter",
        fn=extract_quarter_fn,
        allowed_params={},
        description="Extract quarter (1..4).",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT ISO WEEK NUMBER
# ============================================================

def extract_week_number_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.isocalendar().week
    return _finalize(ctx, col, "weeknum", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_week_number",
        fn=extract_week_number_fn,
        allowed_params={},
        description="Extract ISO week number.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT MONTH START / END
# ============================================================

def extract_is_month_start_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.is_month_start.astype(int)
    return _finalize(ctx, col, "is_month_start", extracted)

def extract_is_month_end_fn(ctx, inputs, params):
    col = inputs[0]
    extracted = pd.to_datetime(ctx.df[col], errors="coerce").dt.is_month_end.astype(int)
    return _finalize(ctx, col, "is_month_end", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_is_month_start",
        fn=extract_is_month_start_fn,
        allowed_params={},
        description="1 if first day of month.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_is_month_end",
        fn=extract_is_month_end_fn,
        allowed_params={},
        description="1 if last day of month.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT QUARTER START / END
# ============================================================

def extract_is_quarter_start_fn(ctx, inputs, params):
    col = inputs[0]
    dt = pd.to_datetime(ctx.df[col], errors="coerce")
    extracted = ((dt.dt.month.isin([1, 4, 7, 10])) & (dt.dt.day == 1)).astype(int)
    return _finalize(ctx, col, "is_quarter_start", extracted)

def extract_is_quarter_end_fn(ctx, inputs, params):
    col = inputs[0]
    dt = pd.to_datetime(ctx.df[col], errors="coerce")
    extracted = ((dt.dt.month.isin([3, 6, 9, 12])) & (dt.dt.is_month_end)).astype(int)
    return _finalize(ctx, col, "is_quarter_end", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_is_quarter_start",
        fn=extract_is_quarter_start_fn,
        allowed_params={},
        description="1 if first day of quarter.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_is_quarter_end",
        fn=extract_is_quarter_end_fn,
        allowed_params={},
        description="1 if last day of quarter.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# EXTRACT TIME DELTA
# ============================================================

def extract_time_delta_fn(ctx, inputs, params):
    col1, col2 = inputs
    s1 = pd.to_datetime(ctx.df[col1], errors="coerce")
    s2 = pd.to_datetime(ctx.df[col2], errors="coerce")
    extracted = (s2 - s1).dt.total_seconds()
    return _finalize(ctx, f"{col2}_minus_{col1}", "seconds", extracted)

TRANSFORM_REGISTRY.register(
    "datetime",
    TransformationDefinition(
        name="extract_time_delta",
        fn=extract_time_delta_fn,
        allowed_params={},
        description="Extract time delta in seconds between two datetime columns.",
        is_derived=True,
        allowed_base=["datetime"],
        output_schema=("numeric", "continuous"),
    )
)
