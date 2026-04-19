# Architecture Documentation

## High‑Level Overview
The system is a four‑layer backend plus a thin Dash UI.  
Each layer is independent, testable, and dataset‑agnostic.

Layers  
1. Data Layer  
2. Data Treatment Layer  
3. Association Layer  
4. Network Layer  
5. API/UI Layer  

---

## 1. Data Layer

### Files
- data/loader.py
- data/schema.py

### Responsibilities
- Load CSV, Parquet, SQL, and API sources.
- Infer column types:
  - numeric (continuous, discrete)
  - categorical (nominal, ordinal)
  - boolean
  - datetime
- Normalize formats and produce a `NormalizedDataFrame`:
  - `.df` — pandas DataFrame
  - `.schema` — dict: column → inferred type
  - `.metadata` — dict: column → descriptive stats

### Design Rules
- Do not mutate original semantics.
- Keep loaders pure and registerable by extension.

---

## 2. Data Treatment Layer

### Files
- data/preprocess.py
- data/policy.py
- data/feature_engineering.py

### Purpose
Handles all **user‑decidable data transformations** that must occur *before* computing associations.

### Responsibilities

#### A. Missing Value Strategy
- drop rows  
- impute (mean, median, mode, KNN)  
- treat missing as a category  

#### B. Outlier Handling
- winsorization  
- clipping  
- log transform  
- leave as‑is  

#### C. Scaling / Normalization
- z‑score  
- min‑max  
- rank transform  
- Box‑Cox / Yeo‑Johnson  

#### D. Encoding Categorical Variables
- label encoding  
- ordinal encoding  
- one‑hot encoding  
- high‑cardinality reduction  

#### E. Numeric Binning
- equal‑width bins  
- quantile bins  
- k‑means bins  
- domain‑specific thresholds  

#### F. Datetime Decomposition
- year, month, day  
- hour  
- weekday/weekend  
- season  
- time deltas  

---

### G. Equational Pre‑Treatment (Multi‑Variable Transformations)

#### 1. Factor Scores (PCA, Factor Analysis)
- Extract latent dimensions from correlated variables.
- Produce composite numeric features.

#### 2. Interaction Terms
- x * y  
- x / y  
- x²  
- x * log(y)

#### 3. Aggregation Across Columns
- sum of symptoms  
- mean of test scores  
- max/min across sensors  

#### 4. Domain‑Specific Composite Features
- BMI  
- financial ratios  
- engagement scores  
- risk indices  

#### 5. Text or Category Embeddings
- TF‑IDF  
- sentence embeddings  
- category embeddings  
- cluster‑based encodings  

---

### Output
`TreatedDataFrame`:
- `.df`  
- `.treatment_policy`  
- `.treatment_log`  

### Design Rules
- All transformations must be logged and reproducible.
- No automatic PCA or factor construction unless explicitly requested.
- Keep transformations pure and deterministic.

---

## 3. Association Layer

### Files
- associations/metrics.py
- associations/engine.py

### Responsibilities
- Provide metric implementations and a registry mapping type pairs to metric functions.
- Auto‑select metric per pair:
  - numeric × numeric → Pearson or Spearman  
  - categorical × categorical → Cramér’s V or Theil’s U  
  - numeric × categorical → correlation ratio or ANOVA  
  - mixed → mutual information or PhiK  
- Compute association matrix in parallel.
- Optional: compute p‑values and apply multiple testing correction.

### Output
`AssociationMatrix`:
- `.matrix`  
- `.metrics_used`  
- `.pvalues`  

### Design Rules
- Metric selection configurable via a policy file or dict.
- Keep metric functions pure and testable.

---

## 4. Network Layer

### Files
- network/builder.py
- network/analysis.py

### Responsibilities
- Convert association matrix to a NetworkX or igraph graph.
- Add node and edge attributes:
  - weight  
  - sign  
  - metric type  
  - p‑value  
- Compute:
  - centrality  
  - clustering  
  - community detection  
- Export adapters:
  - Plotly  
  - PyVis  
  - Cytoscape JSON  

### Output
`NetworkModel`:
- `.graph`  
- `.centrality`  
- `.communities`  

### Design Rules
- Graph construction is stateless and deterministic.
- Visualization adapters separate from analysis logic.

---

## 5. API/UI Layer

### Files
- api/serve.py
- ui/dash_app.py

### Responsibilities
- Expose backend functions for Dash callbacks.
- UI handles:
  - file upload  
  - user controls  
  - rendering  
- Callbacks invoke pure backend functions and pass serializable objects.

### Design Rules
- No heavy computation in callbacks.
- Use caching for expensive steps.
- Keep UI logic minimal and declarative.

---

## Data Flow

1. User uploads dataset or points to a path.  
2. `loader.load` returns DataFrame.  
3. `schema.normalize` returns `NormalizedDataFrame`.  
4. User selects data processing policy. 
5. `preprocess.apply_policy` returns `TreatedDataFrame`.  
6. `engine.compute_associations` returns `AssociationMatrix`.  
7. `builder.build_graph` returns `NetworkModel`.  
8. UI renders graph and exposes export endpoints.
