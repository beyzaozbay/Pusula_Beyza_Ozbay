# src/features/preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Clone-friendly custom transformer (already implemented in your repo)
from src.features.multilabel import MultiLabelBinarizerDF

RAW_PARQ = "data/interim/01_numeric.parquet"
OUT_PARQ = "data/processed/dataset_model_ready.parquet"
OUT_CSV  = "data/processed/dataset_model_ready.csv"
PIPELINE = "models/preprocess_pipeline.joblib"
FEATURES = "reports/feature_names.txt"


def make_ohe():
    """
    Return a OneHotEncoder compatible across scikit-learn versions.

    - scikit-learn >= 1.2 introduces `sparse_output` and deprecates `sparse`.
    - Older versions still use `sparse` (bool).
    We try the modern signature first, then fall back to the legacy one.
    """
    try:
        # New API (>=1.2)
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Legacy API
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    # 0) Load input with already-derived numeric columns
    df0 = pd.read_parquet(RAW_PARQ)

    # 1) Deduplicate:
    #    (a) drop fully identical rows
    #    (b) keep the first record per HastaNo (unique patient row)
    df = df0.drop_duplicates()
    if "HastaNo" in df.columns:
        df = df.drop_duplicates(subset="HastaNo", keep="first")
    df = df.reset_index(drop=True)

    # 2) Column groups
    id_col = "HastaNo"
    target = "TedaviSuresi_num"

    num_cols = ["Yas", "UygulamaSuresi_min"]
    cat_cols = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum", "TedaviAdi"]
    mlb_cols = ["KronikHastalik", "Alerji", "UygulamaYerleri"]
    used_cols = num_cols + cat_cols + mlb_cols

    # 3) Transformers
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])

    mlb = MultiLabelBinarizerDF(columns=mlb_cols)

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ("mlb", mlb, mlb_cols),
        ],
        remainder="drop",
    )

    # 4) Fit/transform on the de-duplicated frame (same source for fit & transform)
    X = df[used_cols]
    Xt = ct.fit_transform(X)
    feat_names = ct.get_feature_names_out()

    # 5) Build a feature DataFrame (dense or sparse-safe)
    if sparse.issparse(Xt):
        # Keep as Pandas Sparse to avoid memory blow-ups if needed
        Xdf = pd.DataFrame.sparse.from_spmatrix(Xt, columns=feat_names)
    else:
        Xdf = pd.DataFrame(Xt, columns=feat_names)

    # 6) Final table: ID + target + features
    base = df[[id_col, target]].reset_index(drop=True)
    Xdf = Xdf.reset_index(drop=True)
    out_df = pd.concat([base, Xdf], axis=1)

    # 7) Safety checks
    assert len(out_df) == len(df), "Row count mismatch after concat."
    assert out_df.columns.duplicated().sum() == 0, "Duplicate column names in final DF."
    if id_col in out_df.columns:
        assert out_df[id_col].nunique() == len(out_df), "HastaNo duplicates remained."

    # 8) Save artifacts
    Path(OUT_PARQ).parent.mkdir(parents=True, exist_ok=True)
    Path(FEATURES).parent.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    out_df.to_parquet(OUT_PARQ, index=False)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    Path(FEATURES).write_text("\n".join(out_df.columns), encoding="utf-8")
    dump(ct, PIPELINE)

    # 9) Logs
    print("transformers:", [name for name, *_ in ct.transformers])
    print("feature_names_out count:", len(feat_names))
    print("✅ Preprocessing completed")
    print(f"Rows (after dedup): {len(out_df)}")
    print(f"Features (X) count: {out_df.shape[1] - 2}")
    print(f"Saved: {OUT_PARQ} and {OUT_CSV}")
    print(f"Pipeline: {PIPELINE}")
    print(f"Feature names: {FEATURES}")


if __name__ == "__main__":
    main()
