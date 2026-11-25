"""Project configuration: official UCI/Kaggle source, clean schema, feature groups, and split settings."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Official dataset sources (UCI ID 697 — identical to Kaggle mirror)
UCI_DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
)
UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
KAGGLE_DATASET_PAGE = (
    "https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention"
)

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # Resulting split: 70% train / 15% val / 15% test (stratified)

# Mapping from raw UCI CSV headers (including trailing tab and typo fixes) to clean snake_case identifiers
RAW_COLUMN_RENAME = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance\t": "daytime_attendance",
    "Daytime/evening attendance": "daytime_attendance",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mothers_qualification",
    "Father's qualification": "fathers_qualification",
    "Mother's occupation": "mothers_occupation",
    "Father's occupation": "fathers_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "educational_special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_fees_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "cu_1st_credited",
    "Curricular units 1st sem (enrolled)": "cu_1st_enrolled",
    "Curricular units 1st sem (evaluations)": "cu_1st_evaluations",
    "Curricular units 1st sem (approved)": "cu_1st_approved",
    "Curricular units 1st sem (grade)": "cu_1st_grade",
    "Curricular units 1st sem (without evaluations)": "cu_1st_without_eval",
    "Curricular units 2nd sem (credited)": "cu_2nd_credited",
    "Curricular units 2nd sem (enrolled)": "cu_2nd_enrolled",
    "Curricular units 2nd sem (evaluations)": "cu_2nd_evaluations",
    "Curricular units 2nd sem (approved)": "cu_2nd_approved",
    "Curricular units 2nd sem (grade)": "cu_2nd_grade",
    "Curricular units 2nd sem (without evaluations)": "cu_2nd_without_eval",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "target",
}

# Feature typing for proper neural network preprocessing (strictly fit on train only)
NOMINAL_CATEGORICAL_COLS = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nationality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
]

BINARY_COLS = [
    "daytime_attendance",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]

ENROLLMENT_NUMERIC_COLS = [
    "application_order",
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]

SEM1_NUMERIC_COLS = [
    "cu_1st_credited",
    "cu_1st_enrolled",
    "cu_1st_evaluations",
    "cu_1st_approved",
    "cu_1st_grade",
    "cu_1st_without_eval",
    "approval_rate_1st_sem",
]

SEM2_NUMERIC_COLS = [
    "cu_2nd_credited",
    "cu_2nd_enrolled",
    "cu_2nd_evaluations",
    "cu_2nd_approved",
    "cu_2nd_grade",
    "cu_2nd_without_eval",
    "approval_rate_2nd_sem",
    "grade_delta_2nd_vs_1st",
    "approval_delta_2nd_vs_1st",
]

# Early-detection stages for practical institutional intervention analysis
STAGE_ENROLLMENT_COLS = NOMINAL_CATEGORICAL_COLS + BINARY_COLS + ENROLLMENT_NUMERIC_COLS
STAGE_SEM1_COLS = STAGE_ENROLLMENT_COLS + SEM1_NUMERIC_COLS
STAGE_FULL_COLS = STAGE_SEM1_COLS + SEM2_NUMERIC_COLS
