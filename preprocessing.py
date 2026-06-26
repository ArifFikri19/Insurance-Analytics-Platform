# ----------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 1: Load + Clean + Basic Preprocessing
# ----------------------------------------

import os
import re
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {
    'Incident Date': 'date',
    'Total Claim Amount': 'numeric',
}

RECOMMENDED_COLUMNS = {
    'Claim ID': 'id',
    'Customer ID': 'id',
    'Age': 'numeric',
    'Months As Customer': 'numeric',
    'Policy Annual Premium': 'numeric',
    'Policy Deductable': 'numeric',
    'Incident Type': 'category',
    'Incident Severity': 'category',
    'Injury Claim': 'numeric',
    'Property Claim': 'numeric',
    'Vehicle Claim': 'numeric',
}

COLUMN_ALIASES = {
    'Claim ID': [
        'claim id', 'claim_id', 'claim number', 'claim no', 'claim reference',
        'claim ref', 'case id', 'case number'
    ],
    'Incident Date': [
        'incident date', 'date of incident', 'accident date', 'claim date',
        'loss date', 'date'
    ],
    'Total Claim Amount': [
        'total claim amount', 'claim amount', 'total claim', 'claim total',
        'amount claimed', 'paid amount', 'claim paid', 'loss amount'
    ],
    'Customer ID': [
        'customer id', 'customer', 'client id', 'insured id', 'policyholder id',
        'member id'
    ],
    'Age': [
        'age', 'insured age', 'customer age', 'client age',
        'custage', 'cust age', 'cust_age', 'customerage'
    ],
    'Months As Customer': [
        'months as customer', 'customer tenure', 'tenure months',
        'months customer', 'months_as_customer'
    ],
    'Policy Annual Premium': [
        'policy annual premium', 'annual premium', 'policy premium',
        'premium amount', 'premium'
    ],
    'Policy Deductable': [
        'policy deductable', 'policy deductible', 'deductible', 'deductable',
        'excess amount'
    ],
    'Incident Type': [
        'incident type', 'accident type', 'claim type', 'loss type'
    ],
    'Incident Severity': [
        'incident severity', 'severity', 'claim severity', 'damage severity'
    ],
    'Injury Claim': [
        'injury claim', 'injury amount', 'bodily injury claim'
    ],
    'Property Claim': [
        'property claim', 'property amount', 'property damage claim'
    ],
    'Vehicle Claim': [
        'vehicle claim', 'vehicle amount', 'auto claim', 'car claim'
    ],
}


def normalize_column_name(name):
    """Normalize a column name for alias matching."""
    return ''.join(ch for ch in str(name).strip().lower() if ch.isalnum())


def load_data(file_path):
    """
    Load Excel or CSV dataset with basic validation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    ext = os.path.splitext(str(file_path).lower())[1]
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV, XLSX, or XLS file.")

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    return df


def standardize_column_names(df):
    """
    Remove leading/trailing spaces from column names and drop blank export columns.
    """
    df.columns = df.columns.str.strip()
    blank_cols = [
        c for c in df.columns
        if (
            not str(c).strip()
            or str(c).lower().startswith('unnamed')
            or re.fullmatch(r'C\d+', str(c).strip())
        )
    ]
    df = df.drop(columns=blank_cols, errors='ignore')
    return df


def detect_column_mapping(df):
    """
    Guess user-uploaded columns for the internal schema.
    Returns a dict of standard column -> uploaded column or None.
    """
    normalized_lookup = {normalize_column_name(c): c for c in df.columns}
    mapping = {}

    for standard_col, aliases in COLUMN_ALIASES.items():
        candidates = [standard_col] + aliases
        match = None
        for candidate in candidates:
            key = normalize_column_name(candidate)
            if key in normalized_lookup:
                match = normalized_lookup[key]
                break
        mapping[standard_col] = match

    return mapping


def standardize_uploaded_columns(df, mapping=None):
    """
    Rename mapped upload columns to the internal schema used by the pipeline.
    """
    df = standardize_column_names(df.copy())
    if mapping is None:
        mapping = detect_column_mapping(df)

    rename_map = {
        source_col: standard_col
        for standard_col, source_col in mapping.items()
        if source_col and source_col in df.columns and source_col != standard_col
    }
    df = df.rename(columns=rename_map)
    return df


def validate_uploaded_dataset(df, mapping=None, max_invalid_pct=0.2):
    """
    Validate uploaded data before running the pipeline.
    Critical errors block processing; warnings are shown but can continue.
    """
    if mapping is not None:
        df = standardize_uploaded_columns(df, mapping)
    else:
        df = standardize_column_names(df.copy())

    errors = []
    warnings = []
    selected_sources = [source for source in (mapping or {}).values() if source]
    duplicated_sources = sorted({source for source in selected_sources if selected_sources.count(source) > 1})
    if duplicated_sources:
        errors.append(
            "The same uploaded column is mapped to multiple fields: "
            + ", ".join(duplicated_sources)
        )

    for col, expected_type in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            continue

        if expected_type == 'numeric':
            values = pd.to_numeric(df[col], errors='coerce')
            invalid_pct = values.isna().mean()
            if invalid_pct > max_invalid_pct:
                errors.append(f"{col} does not look numeric ({invalid_pct:.0%} invalid/missing).")
            elif invalid_pct > 0:
                warnings.append(f"{col} has {invalid_pct:.0%} invalid/missing values; median imputation will be used.")
            if values.dropna().lt(0).any():
                errors.append(f"{col} contains negative values.")

        if expected_type == 'date':
            values = pd.to_datetime(df[col], errors='coerce')
            invalid_pct = values.isna().mean()
            if invalid_pct > max_invalid_pct:
                errors.append(f"{col} does not look like a date ({invalid_pct:.0%} invalid/missing).")
            elif invalid_pct > 0:
                warnings.append(f"{col} has {invalid_pct:.0%} invalid/missing dates.")

    for col, expected_type in RECOMMENDED_COLUMNS.items():
        if col not in df.columns:
            warnings.append(f"Recommended column not mapped/found: {col}")
            continue

        if expected_type == 'numeric':
            values = pd.to_numeric(df[col], errors='coerce')
            invalid_pct = values.isna().mean()
            if invalid_pct > max_invalid_pct:
                warnings.append(f"{col} has many invalid/missing numeric values ({invalid_pct:.0%}).")

    if 'Age' in df.columns:
        ages = pd.to_numeric(df['Age'], errors='coerce').dropna()
        if len(ages) and (ages.lt(0).any() or ages.gt(120).any()):
            warnings.append("Age contains values outside 0-120. Please review the upload.")

    return errors, warnings, df


def convert_dates(df, date_col='Incident Date'):
    """
    Convert date column to datetime. Invalid entries become NaT.
    """
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        missing_dates = df[date_col].isna().sum()
        if missing_dates > 0:
            print(f"Warning: {missing_dates:,} rows have invalid/missing '{date_col}'.")
    return df


def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows. If subset is provided, only consider those columns.
    """
    before = df.shape[0]
    df = df.drop_duplicates(subset=subset)
    after = df.shape[0]
    removed = before - after

    if removed > 0:
        print(f"Removed {removed:,} duplicate rows"
              f"{' (subset: ' + str(subset) + ')' if subset else ''}.")
    else:
        print("No duplicate rows found.")
    return df


def clean_numeric_columns(df, numeric_cols=None):
    """
    Safely coerce important columns to numeric.
    """
    default_numeric_cols = [
        'Total Claim Amount',
        'Injury Claim',
        'Property Claim',
        'Vehicle Claim',
        'Months As Customer',
        'Age',
        'Policy Annual Premium',
        'Policy Deductable',
        'Capital Gains',
        'Capital Loss',
        'Number of Open Complaints'
    ]

    cols_to_clean = numeric_cols if numeric_cols is not None else default_numeric_cols

    for col in cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def handle_missing_values(df, numeric_cols=None, categorical_cols=None):
    """
    Fill missing values intelligently:
      - Datetime columns: left as NaT
      - Numeric columns: median imputation
      - Categorical columns: 'Unknown'
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        id_like = [c for c in numeric_cols if 'id' in c.lower() or 'number' in c.lower()]
        numeric_cols = [c for c in numeric_cols if c not in id_like]

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numeric: median
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical: 'Unknown'
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    return df


def remove_invalid_claims(df):
    """
    Remove rows with negative claim values.
    """
    claim_cols = [
        'Total Claim Amount',
        'Injury Claim',
        'Property Claim',
        'Vehicle Claim'
    ]

    mask = pd.Series(True, index=df.index)
    for col in claim_cols:
        if col in df.columns:
            mask &= df[col] >= 0

    before = len(df)
    df = df[mask]
    removed = before - len(df)

    if removed > 0:
        print(f"Removed {removed:,} rows with negative claim values.")
    else:
        print("No negative claim values found.")
    return df


def basic_time_features(df, date_col='Incident Date'):
    """
    Create Year, Month, Quarter from the date column.
    Safe to run AFTER handle_missing_values because datetime NaTs are handled.
    """
    if date_col not in df.columns:
        print(f"Warning: '{date_col}' not found. Skipping time features.")
        return df

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        print(f"Warning: '{date_col}' is not datetime. Skipping time features.")
        return df

    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Quarter'] = df[date_col].dt.quarter

    return df


def save_cleaned_data(df, output_path="cleaned_insurance_data.csv"):
    """
    Save cleaned dataset.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved: {output_path}")


def preprocessing_pipeline(file_path, dedup_subset=None, column_mapping=None):
    """
    Full preprocessing pipeline for Insurance Quarterly Claim Prediction.

    Returns:
        df: cleaned original dataframe
    """
    print("=" * 50)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 50)

    # 1. Load
    df = load_data(file_path)

    # 2. Clean and standardize column names
    df = standardize_uploaded_columns(df, column_mapping)

    errors, warnings, df = validate_uploaded_dataset(df)
    for warning in warnings:
        print(f"Warning: {warning}")
    if errors:
        raise ValueError("Dataset validation failed: " + " | ".join(errors))

    # 3. Convert dates FIRST (before any imputation)
    df = convert_dates(df, date_col='Incident Date')

    # 4. Remove duplicates
    df = remove_duplicates(df, subset=dedup_subset)

    # 5. Clean numeric columns
    df = clean_numeric_columns(df)

    # 6. Handle missing values (datetime-safe)
    df = handle_missing_values(df)

    # 7. Remove invalid claims
    df = remove_invalid_claims(df)

    # 8. Time features (safe after datetime handling)
    df = basic_time_features(df, date_col='Incident Date')

    # 9. Save
    save_cleaned_data(df)

    print("=" * 50)
    print("PIPELINE COMPLETE")
    print(f"Final cleaned shape: {df.shape}")
    print("=" * 50)

    return df


# ----------------------------------------
# RUN FILE DIRECTLY
# ----------------------------------------
if __name__ == "__main__":
    file_path = "insurance_5years_full_formatted.xlsx"
    df = preprocessing_pipeline(file_path)
    print("\n--- Sample of Cleaned Data ---")
    print(df.head())
