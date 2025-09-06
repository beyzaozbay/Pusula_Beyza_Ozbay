**Name & Surname:** Beyza Özbay  
**Email:** beyza.ozbay.25@gmail.com

# EDA & Preprocessing Summary — Physical Medicine & Rehabilitation

## 1) Dataset & Target
- **Raw dataset:** 2,235 rows × 13 columns (Excel)
- **Target variable:** `TedaviSuresi` (treatment duration); parsed to numeric as `TedaviSuresi_num`
- **ID:** `HastaNo` (patient identifier)

**Columns:**  
`HastaNo, Yas, Cinsiyet, KanGrubu, Uyruk, KronikHastalik, Bolum, Alerji, Tanilar, TedaviAdi, TedaviSuresi, UygulamaYerleri, UygulamaSuresi`

---

## 2) High-level EDA Findings

### 2.1 Schema & Sanity
- **Shape & order:** Matches specification **(2235 × 13)**, exact column names & order.
- **Numeric sanity**
  - `Yas`: non-null in all rows; range **2–92**; no negatives/zeros.
  - `TedaviSuresi` & `UygulamaSuresi`: raw are **textual**, fully parsed downstream (see §3).

### 2.2 Missingness (top)
- `Alerji`: **42.24%**  
- `KanGrubu`: **30.20%**  
- `KronikHastalik`: **27.34%**  
- `UygulamaYerleri`: **9.89%**  
- `Cinsiyet`: **7.56%**  
- `Tanilar`: **3.36%**  
- `Bolum`: **0.49%**  

(See `reports/summary/missingness.csv` and `reports/figures/missingness_bar.png`.)

### 2.3 Duplicates & Patient-level Uniqueness
- **Duplicate full rows:** 928  
- **HastaNo** repeats across many rows in raw.
- Downstream preprocessing enforces **one row per HastaNo** (deduplication in §3).

### 2.4 Main visuals produced
- **Histograms**: `Yas`, `TedaviSuresi_num`, `UygulamaSuresi_min`
- **Boxplots**: Target vs. `Cinsiyet`, target vs. top `Bolum`
- **Scatter**: `Yas` vs. `TedaviSuresi_num`
- **Correlation heatmap**: numeric features
- **Top-count bar charts**: frequent categories / tokens

(See `reports/figures/` for PNGs and `reports/summary/` for CSV summaries.)

---

## 3) Preprocessing Pipeline (Model-readiness)

**Goal:** Clean, consistent, analyzable data; **no modeling required**.

### 3.1 Deduplication
1. Drop exact duplicate rows.  
2. Enforce **one record per `HastaNo`** by keeping the first occurrence.  
**Result:** **404** unique patients.

### 3.2 Derived Numerics
- **`TedaviSuresi_num`**  
  - Parsed from textual `TedaviSuresi` (e.g., “15 Seans”; ranges averaged; robust string parsing).
- **`UygulamaSuresi_min`**  
  - Normalized to **minutes** from `UygulamaSuresi` (supports `dk/saat/sn/gün`).

### 3.3 Imputation
- **Numeric** (`Yas`, `UygulamaSuresi_min`): `SimpleImputer(strategy="median")`
- **Categorical** (`Cinsiyet`, `KanGrubu`, `Uyruk`, `Bolum`, `TedaviAdi`): `SimpleImputer(strategy="most_frequent")`

### 3.4 Encoding & Scaling
- **One-Hot Encoding** for categorical (`OneHotEncoder(handle_unknown="ignore")`)
- **Standardization** for numeric (`StandardScaler`)

### 3.5 Multi-label Features
- Columns: `KronikHastalik`, `Alerji`, `UygulamaYerleri`
- Steps:
  - Tokenization on commas/semicolons; trim/normalize tokens
  - **Binarization** using a clone-safe custom transformer (`MultiLabelBinarizerDF`)
  - Output as additional binary indicator features

### 3.6 Final Feature Matrix
- **ColumnTransformer branches:** `num`, `cat`, `mlb`
- **Feature count (X):** **259** (excludes ID & target)
- **Integrity checks:** no duplicate column names; alignment between pipeline output, processed file, and `reports/feature_names.txt`.

**Artifacts written**
- Processed (ID + target + features):
  - `data/processed/dataset_model_ready.parquet`
  - `data/processed/dataset_model_ready.csv`
- Pipeline: `models/preprocess_pipeline.joblib`
- Feature names: `reports/feature_names.txt`

---

## 4) Reproducibility

### 4.1 End-to-end build
```bash
python -m src.features.derive_numeric
mkdir -p reports/summary
python -m src.visualization.eda_report
python -m src.features.preprocess
4.2 Final single-command validation
bash scripts/final_validate.sh
This script:
Rebuilds all artifacts (derive → EDA → preprocess)
Verifies:
Raw spec (shape + exact columns/order)
Derived numerics presence & non-nullness
EDA outputs (figures + summaries)
Processed dataset integrity (unique HastaNo, expected feature count)
Pipeline correctness (has mlb branch; transform feature count equals processed features; names align)
feature_names.txt equals processed columns; no duplicate names
.gitignore patterns present
README header has Name, Surname & Email near the top