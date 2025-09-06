from pathlib import Path
import pandas as pd
from src.features.parsers import parse_sessions, parse_duration_minutes, to_int_safe

RAW_PATH = Path("data/raw/Talent_Academy_Case_DT_2025.xlsx")
OUT_PARQUET = Path("data/interim/01_numeric.parquet")

def main():
    assert RAW_PATH.exists(), f"Data file not found: {RAW_PATH}"
    df = pd.read_excel(RAW_PATH)

    # Parse target: TedaviSuresi (sessions)
    parsed_sessions = df["TedaviSuresi"].apply(parse_sessions)
    df["TedaviSuresi_num"] = parsed_sessions.apply(to_int_safe)

    # Parse application duration: UygulamaSuresi -> minutes
    parsed_minutes = df["UygulamaSuresi"].apply(parse_duration_minutes)
    df["UygulamaSuresi_min"] = parsed_minutes

    # Basic sanity: non-negative
    for col in ["TedaviSuresi_num", "UygulamaSuresi_min"]:
        if col in df.columns:
            df.loc[df[col].notna() & (df[col] < 0), col] = None

    # Report
    tot = len(df)
    t_nonnull = int(df["TedaviSuresi_num"].notna().sum())
    u_nonnull = int(df["UygulamaSuresi_min"].notna().sum())
    print("=== Derived numeric columns ===")
    print(f"TedaviSuresi_num non-null: {t_nonnull}/{tot}")
    print(f"UygulamaSuresi_min non-null: {u_nonnull}/{tot}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"✅ Saved: {OUT_PARQUET}")

if __name__ == "__main__":
    main()
