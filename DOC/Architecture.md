# Architecture Documentation

## High‑Level Overview

The system is built around a **unified semantic data engine** that executes a user‑defined **DAG (Directed Acyclic Graph)** of steps.  
Each step is a first‑class operation:

- **SchemaCastStep**  
- **TransformStep**  
- **FilterStep**

All steps are defined in registries and executed in topological order.  
The engine is deterministic, extensible, and dataset‑agnostic.

The architecture consists of:

1. **Semantic Data Engine (Unified Layer)**  
2. **Association Layer**  
3. **Network Layer**  
4. **API/UI Layer**

---

# 1. Semantic Data Engine (Unified Layer)

## Purpose

Provide a **registry‑driven, DAG‑based execution model** for all operations that shape or transform the dataset before association analysis.

This layer replaces the former separation between “Data Layer” and “Data Treatment Layer”.  
Schema inference, casting, filtering, and transformations are now **all DAG steps**.

---

## 1.1 Core Concepts

### EngineContext

A context object passed between steps, containing:

- **DataFrame** — the current working dataset  
- **Schema** — semantic type information for each column  
- **Metadata** — descriptive statistics and type‑specific metadata  

The context is treated as immutable: each step produces a new one.

---

### Semantic Schema

Each column has a semantic schema entry describing:

- **base type** (numeric, categorical, boolean, datetime, text, structured, unknown)  
- **subtype** (continuous/discrete, nominal/ordinal, short/long text, etc.)  
- **confidence**  
- **candidate subtypes**  
- **forced flag**  

The schema is used for:

- type‑based filtering  
- transformation validation  
- metadata construction  
- schema casting  

---

## 1.2 DAG Execution Model

### DataPlan

A plan is a list of steps with explicit dependencies.  
The engine sorts steps using topological ordering to ensure correct execution order.

### Step Types

All steps share a common structure:

- **id** — unique identifier  
- **label** — optional UI label  
- **step_type** — `"schema_cast"`, `"transform"`, or `"filter"`  
- **category** — registry category  
- **name** — operation name  
- **inputs** — columns the step depends on  
- **params** — configuration parameters  
- **after** — explicit dependencies on other steps  

The DAG ensures:

- deterministic ordering  
- dependency tracking  
- safe reordering  
- pruning of unused steps  

---

## 1.3 Step Categories

### SchemaCastStep

Casts columns to semantic types using the type registry.  
Examples include numeric casting, datetime parsing, boolean normalization, and text normalization.

### TransformStep

Creates **new columns**.  
Never removes rows or columns.

Transform categories include:

- missing value strategies  
- outlier transformations  
- scaling and normalization  
- encoding  
- binning  
- datetime decomposition  
- composite and multi‑variable features  

### FilterStep

Removes rows or columns.  
Never creates new columns.

Filter categories include:

- value‑based row filtering  
- name‑based column filtering  
- filtering based on semantic
- filtering based on schema types
- invalid values, duplicates  
- variance, cardinality, distribution  
- correlation, multicollinearity  

---

## 1.4 Registries

The engine is fully registry‑driven.

### Type Registry

Defines:

- base types  
- subtypes  
- inference functions  
- metadata builders  
- casting functions  

Used by schema inference and casting steps.

### Transformation Registry

Defines feature‑engineering operations.  
Each transformation returns new columns and may update schema or metadata.

### Filter Registry

Defines row/column filtering operations.  
Each filter returns row masks, column drops, or schema/metadata updates.

---

## 1.5 Execution Order

Steps are executed in topological order:

1. **Schema casting**  
2. **Filtering**  
3. **Transformations**  

This ensures:

- filters run early (filter pushdown)  
- transformations never operate on dropped columns  
- type‑dependent transforms run after schema casting  

---

## 1.6 Output

The Semantic Data Engine produces a final **EngineContext** containing:

- the fully transformed DataFrame  
- the updated semantic schema  
- the updated metadata  

This context is passed to the Association Layer.

---

# 2. Association Layer

## Responsibilities

- Select appropriate metric per type pair  
- Compute association matrix  
- Compute optional p‑values  
- Apply multiple testing correction  
- Execute computations in parallel  

## Output

An **AssociationMatrix** containing:

- association values  
- metrics used  
- optional p‑values  

---

# 3. Network Layer

## Responsibilities

- Convert association matrix into a graph  
- Add node and edge attributes  
- Compute centrality, clustering, and communities  
- Provide export formats for visualization frameworks  

## Output

A **NetworkModel** containing:

- the graph  
- centrality measures  
- community assignments  

---

# 4. API/UI Layer

## Responsibilities

- Expose backend functions to the UI  
- Handle file upload and user controls  
- Render graphs and results  
- Use caching for expensive operations  

---

# Data Flow (Updated)

1. User uploads dataset  
2. Loader returns raw DataFrame  
3. Schema subsystem infers initial schema and metadata  
4. User defines a **DataPlan** (schema casts, filters, transforms)  
5. DAG engine executes steps in topological order  
6. Final EngineContext is produced  
7. Association engine computes associations  
8. Network builder constructs graph  
9. UI renders graph  
