from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.parsers import parse_sessions, parse_duration_minutes, to_int_safe

PARQUET_PATH = Path("data/interim/01_numeric.parquet")
RAW_PATH = Path("data/raw/Talent_Academy_Case_DT_2025.xlsx")
FIG_DIR = Path("reports/figures")
SUM_DIR = Path("reports/summary")

def load_data():
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        print(f"Loaded: {PARQUET_PATH}")
    else:
        assert RAW_PATH.exists(), f"Missing raw data at {RAW_PATH}"
        df = pd.read_excel(RAW_PATH)
        # derive numeric columns inline if parquet is missing
        df["TedaviSuresi_num"] = df["TedaviSuresi"].apply(parse_sessions).apply(to_int_safe)
        df["UygulamaSuresi_min"] = df["UygulamaSuresi"].apply(parse_duration_minutes)
        print(f"Loaded raw and derived numerics: {RAW_PATH}")
    return df

def save_missingness(df):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss_df = miss.reset_index()
    miss_df.columns = ["column", "missing_pct"]
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    miss_df.to_csv(SUM_DIR / "missingness.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=miss_df, x="missing_pct", y="column")
    plt.xlabel("Missing (%)")
    plt.ylabel("Column")
    plt.title("Missingness by Column")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missingness_bar.png", dpi=150)
    plt.close()

def hist_numeric(df, col, bins=30):
    plt.figure(figsize=(8, 5))
    s = pd.to_numeric(df[col], errors="coerce")
    sns.histplot(s.dropna(), bins=bins)
    plt.title(f"Histogram — {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hist_{col}.png", dpi=150)
    plt.close()

def scatter(df, x, y):
    plt.figure(figsize=(7, 6))
    d = df[[x, y]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna()
    sns.scatterplot(data=d, x=x, y=y, s=20)
    plt.title(f"Scatter — {x} vs {y}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"scatter_{x}_vs_{y}.png", dpi=150)
    plt.close()

def corr_heatmap(df, cols):
    plt.figure(figsize=(6, 5))
    c = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = c.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", square=True)
    plt.title("Correlation (numeric)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "corr_heatmap.png", dpi=150)
    plt.close()

def box_by_category(df, cat, y, top_k=None, fname=None):
    d = df[[cat, y]].copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[y])
    if top_k:
        top_vals = d[cat].value_counts().head(top_k).index
        d[cat] = np.where(d[cat].isin(top_vals), d[cat], "Other")
    order = d.groupby(cat)[y].median().sort_values(ascending=False).index
    plt.figure(figsize=(max(7, len(order)*0.6), 6))
    sns.boxplot(data=d, x=cat, y=y, order=order)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"{y} by {cat}")
    plt.tight_layout()
    fname = fname or f"box_{y}_by_{cat}.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()

def top_counts_bar(df, col, top_n=10):
    vc = df[col].value_counts().head(top_n).reset_index()
    vc.columns = [col, "count"]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=vc, x="count", y=col)
    plt.title(f"Top {top_n} — {col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"top_{col}.png", dpi=150)
    plt.close()

def main():
    sns.set_theme()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    rows, cols = df.shape
    pd.DataFrame([{"rows": rows, "cols": cols}]).to_csv(SUM_DIR / "shape.csv", index=False)

    # 1) Missingness
    save_missingness(df)

    # 2) Basic histograms
    for col in ["Yas", "TedaviSuresi_num", "UygulamaSuresi_min"]:
        if col in df.columns:
            hist_numeric(df, col)

    # 3) Scatter & correlation
    if set(["Yas","TedaviSuresi_num"]).issubset(df.columns):
        scatter(df, "Yas", "TedaviSuresi_num")
    numeric_cols = [c for c in ["Yas","TedaviSuresi_num","UygulamaSuresi_min"] if c in df.columns]
    if len(numeric_cols) >= 2:
        corr_heatmap(df, numeric_cols)

    # 4) Boxplots
    if "Cinsiyet" in df.columns and "TedaviSuresi_num" in df.columns:
        box_by_category(df, "Cinsiyet", "TedaviSuresi_num", fname="box_tedavi_by_cinsiyet.png")
    if "Bolum" in df.columns and "TedaviSuresi_num" in df.columns:
        box_by_category(df, "Bolum", "TedaviSuresi_num", top_k=8, fname="box_tedavi_by_bolum_top8.png")

    # 5) Top frequency bars
    for col in ["KanGrubu","Uyruk","Bolum","TedaviAdi"]:
        if col in df.columns:
            top_counts_bar(df, col, top_n=10)

    # 6) Save small numeric summary
    summary = {
        "duplicate_full_rows": int(df.duplicated().sum()),
        "unique_HastaNo": int(df["HastaNo"].nunique()) if "HastaNo" in df.columns else None
    }
    pd.DataFrame([summary]).to_csv(SUM_DIR / "duplicates_summary.csv", index=False)

    print("✅ EDA figures saved to:", FIG_DIR)
    print("✅ Summaries saved to:", SUM_DIR)

if __name__ == "__main__":
    main()
