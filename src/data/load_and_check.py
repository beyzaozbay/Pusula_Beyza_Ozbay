import sys
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("data/raw/Talent_Academy_Case_DT_2025.xlsx")
EXPECTED_ROWS = 2235
EXPECTED_COLS = 13
EXPECTED_COL_NAMES = [
    "HastaNo","Yas","Cinsiyet","KanGrubu","Uyruk",
    "KronikHastalik","Bolum","Alerji","Tanilar","TedaviAdi",
    "TedaviSuresi","UygulamaYerleri","UygulamaSuresi"
]

def main():
    if not DATA_PATH.exists():
        print(f"❌ Data file not found at: {DATA_PATH}")
        sys.exit(1)

    # Load
    try:
        df = pd.read_excel(DATA_PATH)
    except Exception as e:
        print("❌ Failed to read Excel:", e)
        sys.exit(1)

    # Shape check
    rows, cols = df.shape
    ok_shape = (rows == EXPECTED_ROWS and cols == EXPECTED_COLS)

    # Column-name check (set equality; order-insensitive)
    col_list = list(df.columns)
    set_match = set(col_list) == set(EXPECTED_COL_NAMES)
    order_match = (col_list == EXPECTED_COL_NAMES)

    # Report basic info
    print("=== Overview ===")
    print(f"Path        : {DATA_PATH}")
    print(f"Shape       : {rows} rows x {cols} cols")
    print(f"Cols (found): {col_list}")
    print(f"Match shape (2235x13): {ok_shape}")
    print(f"Match column names (set): {set_match}")
    print(f"Match column order     : {order_match}")
    missing = list(set(EXPECTED_COL_NAMES) - set(col_list))
    extra   = list(set(col_list) - set(EXPECTED_COL_NAMES))
    if missing: print("Missing columns:", missing)
    if extra:   print("Extra columns  :", extra)

    # Decide pass/fail on hard requirements
    if not ok_shape or not set_match:
        print("❌ Hard requirement failed (shape and/or required column names).")
        sys.exit(1)

    # Duplicates
    dup_rows = int(df.duplicated().sum())
    print("\n=== Duplicates ===")
    print(f"Duplicate full rows: {dup_rows}")
    if "HastaNo" in df.columns:
        dup_hastano = int(df.duplicated(subset=["HastaNo"]).sum())
        uniq_hastano = int(df["HastaNo"].nunique(dropna=True))
        print(f"Unique HastaNo: {uniq_hastano} | Duplicated HastaNo: {dup_hastano}")

    # Missingness
    na_counts = df.isna().sum()
    na_pct = (na_counts / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"missing_count": na_counts, "missing_pct": na_pct})
    print("\n=== Missingness (top 10 by pct) ===")
    print(miss_df.sort_values("missing_pct", ascending=False).head(10).to_string())

    # Numeric sanity checks (coerce)
    numeric_targets = ["Yas","TedaviSuresi","UygulamaSuresi"]
    print("\n=== Numeric sanity ===")
    for c in numeric_targets:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            print(f"- {c}: non-null={s.notna().sum()}, min={s.min()}, max={s.max()}, negatives={(s<0).sum()}, zeros={(s==0).sum()}, non_numeric_or_missing={s.isna().sum()}")

    # Soft warnings for order mismatch
    if not order_match:
        print("\n⚠️ Column order differs from the spec; this is not critical but keep it consistent if possible.")

    print("\n✅ All required case checks satisfied. Data is ready for EDA.")
    sys.exit(0)

if __name__ == "__main__":
    main()
