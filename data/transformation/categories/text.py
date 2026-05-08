import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data.transformation.registry import (
    TRANSFORM_REGISTRY,
    TransformationDefinition,
    TransformationResult,
)


# ============================================================
# LOWERCASE
# ============================================================

def text_lower_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].astype(str).str.lower()

    new_col = f"{col}_lower"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="lowercase",
        fn=text_lower_fn,
        allowed_params={},
        description="Convert text to lowercase.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("text", "short_text"),
    )
)


# ============================================================
# UPPERCASE
# ============================================================

def text_upper_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].astype(str).str.upper()

    new_col = f"{col}_upper"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="uppercase",
        fn=text_upper_fn,
        allowed_params={},
        description="Convert text to uppercase.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("text", "short_text"),
    )
)


# ============================================================
# STRIP WHITESPACE
# ============================================================

def text_strip_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].astype(str).str.strip()

    new_col = f"{col}_strip"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="strip",
        fn=text_strip_fn,
        allowed_params={},
        description="Strip leading and trailing whitespace.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("text", "short_text"),
    )
)


# ============================================================
# REMOVE PUNCTUATION
# ============================================================

def text_remove_punct_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].astype(str).str.replace(r"[^\w\s]", "", regex=True)

    new_col = f"{col}_nopunct"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="remove_punctuation",
        fn=text_remove_punct_fn,
        allowed_params={},
        description="Remove punctuation from text.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("text", "short_text"),
    )
)


# ============================================================
# REGEX EXTRACT
# ============================================================

def text_regex_extract_fn(ctx, inputs, params):
    col = inputs[0]
    pattern = params["pattern"]

    s = ctx.df[col].astype(str).str.extract(pattern, expand=False)

    new_col = f"{col}_regex"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="regex_extract",
        fn=text_regex_extract_fn,
        allowed_params={"pattern": str},
        description="Extract first regex group from text.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("text", "short_text"),
    )
)


# ============================================================
# LENGTH
# ============================================================

def text_length_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].astype(str).str.len()

    new_col = f"{col}_length"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="length",
        fn=text_length_fn,
        allowed_params={},
        description="Compute text length in characters.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# WORD COUNT
# ============================================================

def text_word_count_fn(ctx, inputs, params):
    col = inputs[0]
    s = ctx.df[col].astype(str).str.split().str.len()

    new_col = f"{col}_wordcount"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="word_count",
        fn=text_word_count_fn,
        allowed_params={},
        description="Count number of words in text.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# CONTAINS PATTERN
# ============================================================

def text_contains_fn(ctx, inputs, params):
    col = inputs[0]
    pattern = params["pattern"]

    s = ctx.df[col].astype(str).str.contains(pattern, regex=True, na=False).astype(int)

    new_col = f"{col}_contains"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="contains",
        fn=text_contains_fn,
        allowed_params={"pattern": str},
        description="Return 1 if text matches regex pattern, else 0.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("numeric", "discrete"),
    )
)


# ============================================================
# REPLACE PATTERN
# ============================================================

def text_replace_fn(ctx, inputs, params):
    col = inputs[0]
    pattern = params["pattern"]
    repl = params.get("replacement", "")

    s = ctx.df[col].astype(str).str.replace(pattern, repl, regex=True)

    new_col = f"{col}_replace"
    return TransformationResult(
        new_columns={new_col: pd.Series(s, index=ctx.df.index, name=new_col)},
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="replace",
        fn=text_replace_fn,
        allowed_params={"pattern": str, "replacement": str},
        description="Replace regex pattern in text.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("text", "short_text"),
    )
)


# ============================================================
# TF-IDF TRANSFORM
# ============================================================

def tfidf_fn(ctx, inputs, params):
    col = inputs[0]
    series = ctx.df[col].astype(str)

    vocab = params.get("vocabulary", None)

    vectorizer = TfidfVectorizer(
        max_features=params.get("max_features", None),
        ngram_range=params.get("ngram_range", (1, 1)),
        stop_words=params.get("stop_words", None),
        vocabulary=vocab,
    )

    matrix = vectorizer.fit_transform(series)  

    feature_names = vectorizer.get_feature_names_out()

    new_cols = {}
    for i, name in enumerate(feature_names):
        col_name = f"{col}_tfidf_{name}"
        values = matrix.getcol(i).toarray().ravel()
        new_cols[col_name] = pd.Series(values, index=ctx.df.index, name=col_name)

    return TransformationResult(
        new_columns=new_cols,
        terminal=True,
    )



TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="tfidf",
        fn=tfidf_fn,
        allowed_params={
            "max_features": int,
            "ngram_range": tuple,
            "stop_words": (list, str, type(None)),
            "vocabulary": list,
        },
        description="Compute TF-IDF vectorization of a text column.",
        is_derived=False,
        allowed_base=["text"],
        output_schema=("numeric", "continuous"),
    )
)

# ============================================================
# REMOVE STOPWORDS
# ============================================================

def remove_stopwords_fn(ctx, inputs, params):
    col = inputs[0]
    series = ctx.df[col].astype(str)

    # User-provided stopwords (list of strings)
    stopwords = set(params.get("stopwords", []))

    # Tokenize on whitespace
    tokens = series.str.split()

    # Remove stopwords
    cleaned = tokens.apply(
        lambda words: " ".join(w for w in words if w.lower() not in stopwords)
    )

    new_col = f"{col}_nostop"
    return TransformationResult(
        new_columns={
            new_col: pd.Series(cleaned, index=ctx.df.index, name=new_col)
        },
        terminal=True,
    )


TRANSFORM_REGISTRY.register(
    "text",
    TransformationDefinition(
        name="remove_stopwords",
        fn=remove_stopwords_fn,
        allowed_params={"stopwords": list},
        description="Remove stopwords from a text column.",
        is_derived=True,
        allowed_base=["text"],
        output_schema=("text", "short_text"),
    )
)