#!/usr/bin/env bash
set -euo pipefail

echo "=== FINAL FULL REBUILD & VALIDATION START ==="

# 0) Rebuild artifacts (idempotent)
mkdir -p reports/summary
python -m src.features.derive_numeric
python -m src.visualization.eda_report
python -m src.features.preprocess

# 1) Strict validation (everything required by the case) — NO README HEADER CHECK
python - <<'PY'
import sys, pathlib
import pandas as pd, numpy as np, joblib

FAIL = []

RAW_XLSX = "data/raw/Talent_Academy_Case_DT_2025.xlsx"
NUM_PARQ = "data/interim/01_numeric.parquet"
OUT_PARQ = "data/processed/dataset_model_ready.parquet"
PIPE     = "models/preprocess_pipeline.joblib"
FEATS_TXT= "reports/feature_names.txt"

# 1A) Raw spec: exact shape & column order
try:
    raw = pd.read_excel(RAW_XLSX)
    expected_cols = ['HastaNo','Yas','Cinsiyet','KanGrubu','Uyruk',
                     'KronikHastalik','Bolum','Alerji','Tanilar','TedaviAdi',
                     'TedaviSuresi','UygulamaYerleri','UygulamaSuresi']
    if raw.shape != (2235, 13):
        FAIL.append(f"Raw shape mismatch: {raw.shape} (expected 2235x13)")
    if list(raw.columns) != expected_cols:
        FAIL.append("Raw columns/order mismatch.")
    else:
        print("✅ raw spec OK")
except Exception as e:
    FAIL.append(f"Raw read error: {e}")

# 1B) Derived numerics present & fully non-null
try:
    num = pd.read_parquet(NUM_PARQ)
    if num['TedaviSuresi_num'].notna().sum() != 2235:
        FAIL.append("TedaviSuresi_num non-null count != 2235")
    if num['UygulamaSuresi_min'].notna().sum() != 2235:
        FAIL.append("UygulamaSuresi_min non-null count != 2235")
    else:
        print("✅ numeric derivations OK")
except Exception as e:
    FAIL.append(f"Numeric parquet error: {e}")

# 1C) EDA outputs exist (at least one figure + one summary)
from pathlib import Path
eda_ok = Path("reports/summary/shape.csv").exists() and Path("reports/figures/corr_heatmap.png").exists()
print("✅ EDA outputs OK" if eda_ok else "❌ EDA outputs missing")
if not eda_ok:
    FAIL.append("EDA outputs missing (summary/figures)")

# 1D) Processed dataset: single row per HastaNo
try:
    proc = pd.read_parquet(OUT_PARQ)
    rows = len(proc)
    uniq = proc['HastaNo'].nunique()
    dup_ids = proc['HastaNo'].duplicated().sum()
    cols = proc.shape[1]
    feats_in_file = cols - 2  # drop ID + target
    print(f"processed: rows={rows} | unique HastaNo={uniq} | dup ids={dup_ids} | total cols={cols} => features={feats_in_file}")
    if rows != uniq or dup_ids != 0:
        FAIL.append("Processed dataset NOT CLEAN: HastaNo not unique.")
    else:
        print("✅ processed CLEAN (HastaNo unique)")
except Exception as e:
    FAIL.append(f"Processed parquet error: {e}")

# 1E) Pipeline: has 'mlb' branch, transform shape == processed features, and names align
try:
    pipe = joblib.load(PIPE)
    names_map = getattr(pipe, "named_transformers_", {})
    has_mlb = "mlb" in names_map
    X_input_cols = ['Yas','UygulamaSuresi_min','Cinsiyet','KanGrubu','Uyruk',
                    'Bolum','TedaviAdi','KronikHastalik','Alerji','UygulamaYerleri']
    num = pd.read_parquet(NUM_PARQ)  # ensure 'num' is available even if 1B failed
    Xt = pipe.transform(num[X_input_cols].head(10))
    arr = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    feat_names = pipe.get_feature_names_out()
    ok = has_mlb and (arr.shape[1] == len(feat_names) == (proc.shape[1]-2))
    print(f"transformers: {list(names_map.keys()) if names_map else 'unknown'}")
    print(f"transform features: {arr.shape[1]} | expected: {proc.shape[1]-2} | mlb: {has_mlb}")
    print("✅ pipeline OK" if ok else "❌ pipeline mismatch")
    if not ok:
        FAIL.append("Pipeline feature count/name mismatch (or 'mlb' missing).")
except Exception as e:
    FAIL.append(f"Pipeline load/transform error: {e}")

# 1F) feature_names.txt matches processed columns; no duplicate column names
try:
    feats_txt = pathlib.Path(FEATS_TXT).read_text(encoding="utf-8").splitlines()
    dup_cols = proc.columns[proc.columns.duplicated()].tolist()
    if feats_txt != list(proc.columns):
        FAIL.append("feature_names.txt != processed df.columns")
    if dup_cols:
        FAIL.append(f"Duplicate columns in processed file: {dup_cols}")
    else:
        print(f"✅ feature_names.txt OK ({len(feats_txt)} names), duplicates: {len(dup_cols)}")
except Exception as e:
    FAIL.append(f"feature_names.txt error: {e}")

# 1G) .gitignore patterns present (simple substring check)
try:
    gi_need = [".venv/","__pycache__/","*.pyc","data/raw/*","data/interim/*","data/processed/*"]
    gi_txt = pathlib.Path(".gitignore").read_text(encoding="utf-8")
    for pat in gi_need:
        if pat not in gi_txt:
            FAIL.append(f".gitignore missing pattern: {pat}")
    if not any(msg.startswith(".gitignore") for msg in FAIL):
        print("✅ .gitignore OK")
except Exception as e:
    FAIL.append(f".gitignore check error: {e}")

# 1H) Optional documentation presence — check reports/DOCUMENTATION.md
doc_path = pathlib.Path("reports/DOCUMENTATION.md")
if doc_path.exists():
    print("✅ Optional EDA/preprocessing summary exists (reports/DOCUMENTATION.md)")
else:
    print("ℹ️ Optional summary doc not found at reports/DOCUMENTATION.md (optional)")

# Final
if FAIL:
    print("\n❌ VALIDATION FAILED")
    for m in FAIL: print(" -", m)
    sys.exit(1)
else:
    print("\n✅ ALL TESTS PASSED — repository is ready.")
    sys.exit(0)
PY

echo "=== FINAL FULL REBUILD & VALIDATION DONE ==="
