"""Download, audit, clean, and prepare the official Student Dropout dataset (UCI ID 697 / Kaggle)."""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (
    BINARY_COLS,
    KAGGLE_DATASET_PAGE,
    NOMINAL_CATEGORICAL_COLS,
    PROCESSED_DIR,
    RAW_COLUMN_RENAME,
    RAW_DIR,
    STAGE_FULL_COLS,
    UCI_DATASET_PAGE,
    UCI_DOWNLOAD_URL,
)


def download_raw_dataset(force: bool = False) -> pd.DataFrame:
    """Download the official ZIP from UCI Machine Learning Repository if not already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = RAW_DIR / "student_dropout_raw.csv"
    if not raw_csv.exists() or force:
        print(f"[UCI] Downloading dataset from {UCI_DOWNLOAD_URL}...")
        req = urllib.request.Request(UCI_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw_bytes = urllib.request.urlopen(req, timeout=25).read()
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            raw_csv.write_bytes(zf.read(csv_name))
        print(f"      Saved {raw_csv.name} ({raw_csv.stat().st_size / 1e3:.1f} KB)")
    return pd.read_csv(raw_csv, sep=";")


def engineer_academic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable academic performance ratios (zero leakage: strictly within student record)."""
    out = df.copy()
    out["approval_rate_1st_sem"] = np.where(
        out["cu_1st_enrolled"] > 0,
        (out["cu_1st_approved"] / out["cu_1st_enrolled"]).clip(0.0, 1.0),
        0.0,
    )
    out["approval_rate_2nd_sem"] = np.where(
        out["cu_2nd_enrolled"] > 0,
        (out["cu_2nd_approved"] / out["cu_2nd_enrolled"]).clip(0.0, 1.0),
        0.0,
    )
    out["grade_delta_2nd_vs_1st"] = out["cu_2nd_grade"] - out["cu_1st_grade"]
    out["approval_delta_2nd_vs_1st"] = out["approval_rate_2nd_sem"] - out["approval_rate_1st_sem"]
    return out


def prepare_datasets(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit raw data, normalize column names, engineer ratios, and save processed CSVs + metadata."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = download_raw_dataset(force=force_download)

    # Audit raw integrity
    n_nulls = int(raw_df.isna().sum().sum())
    n_duplicates = int(raw_df.duplicated().sum())

    # Rename columns cleanly
    df = raw_df.rename(columns=RAW_COLUMN_RENAME).copy()
    df = engineer_academic_ratios(df)

    # Target encodings:
    # 1) All 4,424 students: binary target_dropout (1 = Dropout, 0 = Graduate or Enrolled) + multiclass target
    df_all = df.copy()
    df_all["target_dropout"] = (df_all["target"] == "Dropout").astype(int)

    # 2) Final-outcome students only (3,630 students): 1 = Dropout, 0 = Graduate (excludes ambiguous Enrolled)
    df_binary = df[df["target"].isin(["Dropout", "Graduate"])].copy().reset_index(drop=True)
    df_binary["target_dropout"] = (df_binary["target"] == "Dropout").astype(int)

    all_path = PROCESSED_DIR / "student_dropout_all.csv"
    bin_path = PROCESSED_DIR / "student_dropout_binary.csv"
    df_all.to_csv(all_path, index=False)
    df_binary.to_csv(bin_path, index=False)

    # Descriptive EDA summary on dropout rates by key student profiles (all 4,424 students)
    eda_dropout_rates = {
        "overall_all_pct": round(float(df_all["target_dropout"].mean() * 100), 2),
        "overall_binary_final_outcome_pct": round(float(df_binary["target_dropout"].mean() * 100), 2),
        "tuition_up_to_date_yes_pct": round(float(df_all.loc[df_all["tuition_fees_up_to_date"] == 1, "target_dropout"].mean() * 100), 2),
        "tuition_up_to_date_no_pct": round(float(df_all.loc[df_all["tuition_fees_up_to_date"] == 0, "target_dropout"].mean() * 100), 2),
        "scholarship_yes_pct": round(float(df_all.loc[df_all["scholarship_holder"] == 1, "target_dropout"].mean() * 100), 2),
        "scholarship_no_pct": round(float(df_all.loc[df_all["scholarship_holder"] == 0, "target_dropout"].mean() * 100), 2),
        "debtor_yes_pct": round(float(df_all.loc[df_all["debtor"] == 1, "target_dropout"].mean() * 100), 2),
        "debtor_no_pct": round(float(df_all.loc[df_all["debtor"] == 0, "target_dropout"].mean() * 100), 2),
        "attendance_daytime_pct": round(float(df_all.loc[df_all["daytime_attendance"] == 1, "target_dropout"].mean() * 100), 2),
        "attendance_evening_pct": round(float(df_all.loc[df_all["daytime_attendance"] == 0, "target_dropout"].mean() * 100), 2),
        "mean_approval_rate_1st_sem_dropout": round(float(df_binary.loc[df_binary["target_dropout"] == 1, "approval_rate_1st_sem"].mean() * 100), 1),
        "mean_approval_rate_1st_sem_graduate": round(float(df_binary.loc[df_binary["target_dropout"] == 0, "approval_rate_1st_sem"].mean() * 100), 1),
        "mean_approval_rate_2nd_sem_dropout": round(float(df_binary.loc[df_binary["target_dropout"] == 1, "approval_rate_2nd_sem"].mean() * 100), 1),
        "mean_approval_rate_2nd_sem_graduate": round(float(df_binary.loc[df_binary["target_dropout"] == 0, "approval_rate_2nd_sem"].mean() * 100), 1),
    }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "uci_page": UCI_DATASET_PAGE,
            "kaggle_page": KAGGLE_DATASET_PAGE,
            "citation": (
                "Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). "
                "Predicting Student Dropout and Academic Success. Data, 7(11), 146."
            ),
            "institution": "Instituto Politécnico de Portalegre (Portugal), 17 cursos de graduação (2008–2019)",
        },
        "audit": {
            "raw_rows": int(len(raw_df)),
            "raw_feature_cols": int(raw_df.shape[1] - 1),
            "engineered_feature_cols": 4,
            "total_model_input_raw_cols": len(STAGE_FULL_COLS),
            "missing_values_total": n_nulls,
            "exact_duplicate_rows": n_duplicates,
            "nominal_categorical_cols": len(NOMINAL_CATEGORICAL_COLS),
            "binary_cols": len(BINARY_COLS),
        },
        "class_distribution": {
            k: int(v) for k, v in df_all["target"].value_counts().items()
        },
        "processed_counts": {
            "all_students_rows": int(len(df_all)),
            "binary_final_outcome_rows": int(len(df_binary)),
            "excluded_ambiguous_enrolled_in_binary": int((df_all["target"] == "Enrolled").sum()),
        },
        "eda_dropout_rates": eda_dropout_rates,
    }

    meta_path = PROCESSED_DIR / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Prepared] {all_path.name}: {len(df_all):,} rows | {bin_path.name}: {len(df_binary):,} rows")
    print(f"[Audit]    Nulls={n_nulls} | Duplicates={n_duplicates} | Classes={metadata['class_distribution']}")
    print(f"[EDA]      Dropout rate (tuition overdue vs up-to-date): {eda_dropout_rates['tuition_up_to_date_no_pct']}% vs {eda_dropout_rates['tuition_up_to_date_yes_pct']}%")
    print(f"[EDA]      1st-sem approval rate (Dropout vs Graduate):  {eda_dropout_rates['mean_approval_rate_1st_sem_dropout']}% vs {eda_dropout_rates['mean_approval_rate_1st_sem_graduate']}%")
    return df_all, df_binary, metadata


if __name__ == "__main__":
    prepare_datasets()
