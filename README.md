**Name & Surname:** Beyza Özbay  
**Email:** beyza.ozbay.25@gmail.com

# Data Science Intern Case Study — Physical Medicine & Rehabilitation

## Overview
**Dataset:** Physical Medicine & Rehabilitation — **2235 rows**, **13 columns**  
**Target:** `TedaviSuresi` (treatment duration in sessions)  
**Goal:** Perform **in-depth EDA** and make the data **model-ready** (clean, consistent, analyzable). **No modeling required.**

### Columns
- `HastaNo` — Anonymized patient ID  
- `Yas` — Age  
- `Cinsiyet` — Gender  
- `KanGrubu` — Blood type  
- `Uyruk` — Nationality  
- `KronikHastalik` — Chronic conditions (comma/semicolon separated)  
- `Bolum` — Department/Clinic  
- `Alerji` — Allergies (may be single or multi-value)  
- `Tanilar` — Diagnoses  
- `TedaviAdi` — Treatment name  
- **`TedaviSuresi`** — Treatment duration in sessions (**target**)  
- `UygulamaYerleri` — Application sites  
- `UygulamaSuresi` — Application duration

---

## Requirements
- Python **3.9+**

**Packages** (in `requirements.txt`):  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `openpyxl`, `pyarrow`, `jupyter`, `ipykernel`

> Note: Uses `OneHotEncoder(handle_unknown="ignore")` (no `sparse` arg). If you pin versions, **`scikit-learn>=1.2`** is safe.

---

## Project Structure
```
.
├── data/
│   ├── raw/          # original data (not committed)
│   ├── interim/      # derived numerics / temp
│   └── processed/    # model-ready dataset
├── notebooks/        # (optional) exploratory notebooks
├── src/
│   ├── data/         # data loading & checks
│   ├── features/     # parsers, multi-label, preprocessing
│   └── visualization/ # EDA report scripts
├── reports/
│   ├── figures/      # plots (hist, scatter, heatmap, box, top counts)
│   └── summary/      # csv summaries
├── models/           # saved preprocessing pipeline
├── docs/             # REQUIREMENTS.md, DATA_DICTIONARY.md (optional)
├── scripts/
│   └── final_validate.sh  # single authoritative validator (rebuild + checks)
└── README.md
```

---

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Data
Place the provided Excel file at:
```
data/raw/Talent_Academy_Case_DT_2025.xlsx
```

---

## How to Run (Step-by-step)

### 1) Load & spec check
Validates shape (2235×13), column names/order, duplicates, and basic missingness.
```bash
python -m src.data.load_and_check
```

### 2) Derive numeric columns
- `TedaviSuresi_num`: parsed from textual `TedaviSuresi` (e.g., “15 Seans”; ranges averaged)  
- `UygulamaSuresi_min`: minutes from `UygulamaSuresi` (dk/saat/sn/gün)
```bash
python -m src.features.derive_numeric
```

### 3) EDA figures & summaries
Generates histograms, scatter, correlation heatmap, missingness bar, boxplots, top-count bars.  
Outputs to `reports/figures/` and CSV summaries to `reports/summary/`.
```bash
mkdir -p reports/summary
python -m src.visualization.eda_report
```

### 4) Preprocessing → model-ready dataset
- Dedup (exact dups, then by `HastaNo`, keep first)  
- Imputation (numeric=median, categorical=most_frequent)  
- One-Hot Encoding for categorical  
- Standard scaling for numeric  
- Multi-label binarization for `KronikHastalik`, `Alerji`, `UygulamaYerleri`
```bash
python -m src.features.preprocess
```

---

## Outputs

### Figures (`reports/figures/`)
- `missingness_bar.png`
- `hist_Yas.png`
- `hist_TedaviSuresi_num.png`
- `hist_UygulamaSuresi_min.png`
- `scatter_Yas_vs_TedaviSuresi_num.png`
- `corr_heatmap.png`
- `box_tedavi_by_cinsiyet.png`
- `box_tedavi_by_bolum_top8.png`
- `top_*.png`

### Summaries (`reports/summary/`)
- `missingness.csv`
- `shape.csv`
- `duplicates_summary.csv`

### Processed dataset (`data/processed/`)
- `dataset_model_ready.parquet`
- `dataset_model_ready.csv`

### Pipeline & Features
- `models/preprocess_pipeline.joblib`  
- `reports/feature_names.txt` (ID, target, and all feature columns)

---

## Final Validation (Single Command)
Clean rebuild and validate all required items (raw spec, derived numerics, EDA outputs, processed dataset integrity, pipeline with multi-label branch & feature counts, feature names alignment, `.gitignore`, and README header):
```bash
bash scripts/final_validate.sh
```

---

## Notes
- Multi-label fields (`KronikHastalik`, `Alerji`, `UygulamaYerleri`) are tokenized (comma/semicolon) and binarized.  
- `TedaviSuresi_num` is integer sessions; `UygulamaSuresi_min` is minutes.  
- Duplicates are removed to ensure one clean row per `HastaNo`.

---

## Submission
- **GitHub repo name:** `Pusula_Beyza_Ozbay`  
- Include this `README.md` at the repo root (Name, Surname, Email at the top).  
- *(Optional)* Include your EDA/preprocessing summary under `reports/` with Name, Surname, Email at the top.  
- **Deadline:** 06.09.2025 23:59
