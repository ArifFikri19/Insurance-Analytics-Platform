# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 4: Train-Test Split (Time-Series Aware)
# Uses quarterly dataset from aggregation.py
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np


FORECAST_SAFE_FEATURES = [
    'Quarterly_Total_Claims_Lag1',
    'Quarterly_Total_Claims_Lag2',
    'Quarterly_Total_Claims_Lag3',
    'Quarterly_Total_Claims_Lag4',
    'Quarterly_Claim_Count_Lag1',
    'Quarterly_Claim_Count_Lag2',
    'Quarterly_Claim_Count_Lag3',
    'Quarterly_Claim_Count_Lag4',
    'Quarterly_Avg_Claim_Lag1',
    'Quarterly_Avg_Claim_Lag2',
    'Quarterly_Avg_Claim_Lag3',
    'Quarterly_Avg_Claim_Lag4',
    'Quarterly_Total_Claims_Rolling_Mean_4Q',
    'Quarterly_Total_Claims_Rolling_Std_4Q',
    'Quarter_Sin',
    'Quarter_Cos',
    'Q_1',
    'Q_2',
    'Q_3',
    'Q_4',
    'Time_Trend',
]


# ---------------------------------------------------------
# LOAD QUARTERLY DATA
# ---------------------------------------------------------
def load_quarterly_data(file_path="quarterly_claims.csv"):
    """
    Load quarterly aggregated dataset from step 3.
    Checks multiple possible locations.
    """
    # Check given path first
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Quarterly dataset loaded from: {file_path}")
        print(f"Shape: {df.shape}")
        return df

    # Try outputs/ folder
    alt_path = os.path.join("outputs", os.path.basename(file_path))
    if os.path.exists(alt_path):
        df = pd.read_csv(alt_path)
        print(f"Quarterly dataset loaded from: {alt_path}")
        print(f"Shape: {df.shape}")
        return df

    # Try other common locations
    for path in ["outputs/quarterly_claims.csv", "outputs/aggregated_data.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Quarterly dataset loaded from: {path}")
            print(f"Shape: {df.shape}")
            return df

    raise FileNotFoundError(f"Quarterly dataset not found: {file_path}")

# ---------------------------------------------------------
# IDENTIFY FEATURES AND TARGET
# ---------------------------------------------------------
def get_feature_columns(df, target_col='Quarterly_Total_Claims', drop_cols=None):
    """
    Identify forecast-safe feature columns.

    Current-quarter claim statistics and current-quarter growth values are useful
    for historical analysis, but they are not known before a future quarter
    happens. Training only on lag/rolling/seasonal/trend features prevents
    leakage and gives a more honest forecasting model.
    """
    feature_cols = [c for c in FORECAST_SAFE_FEATURES if c in df.columns]
    if not feature_cols:
        if drop_cols is None:
            drop_cols = [
                target_col,
                'Quarter_Label',
                'Year',
                'Quarter'
            ]
        feature_cols = [c for c in df.columns if c not in drop_cols]

    print(f"Target: {target_col}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    return feature_cols


# ---------------------------------------------------------
# TIME-BASED SPLIT (configurable)
# ---------------------------------------------------------
def time_based_split(df, train_ratio=0.8):
    """
    Split data chronologically for time-series.
    First N% for training, remainder for testing.
    Preserves temporal order.
    """
    n = len(df)
    split_idx = int(n * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Time-based split ({train_ratio*100:.0f}/{100-train_ratio*100:.0f}): {len(train_df)} train | {len(test_df)} test")
    print(f"Train range: {train_df['Quarter_Label'].iloc[0]} to {train_df['Quarter_Label'].iloc[-1]}")
    print(f"Test range:  {test_df['Quarter_Label'].iloc[0]} to {test_df['Quarter_Label'].iloc[-1]}")

    return train_df, test_df


# ---------------------------------------------------------
# THREE SPLIT VERSIONS
# ---------------------------------------------------------
def split_70_30(df):
    """70/30 train-test split."""
    return time_based_split(df, train_ratio=0.7)


def split_80_20(df):
    """80/20 train-test split."""
    return time_based_split(df, train_ratio=0.8)


def split_90_10(df):
    """90/10 train-test split."""
    return time_based_split(df, train_ratio=0.9)


# ---------------------------------------------------------
# EXTRACT X AND Y
# ---------------------------------------------------------
def extract_xy(df, feature_cols, target_col='Quarterly_Total_Claims'):
    """
    Extract X (features) and y (target) from dataframe.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


# ---------------------------------------------------------
# SAVE SPLITS
# ---------------------------------------------------------
def save_splits(train_df, test_df,
                train_path="train_data.csv",
                test_path="test_data.csv",
                feature_cols=None):
    """
    Save train and test datasets.
    Also saves feature list for later use.
    """
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train data saved: {train_path} ({len(train_df)} rows)")
    print(f"Test data saved:  {test_path} ({len(test_df)} rows)")

    # Save feature columns list
    if feature_cols:
        with open("feature_columns.txt", "w") as f:
            f.write("\n".join(feature_cols))
        print("Feature columns saved: feature_columns.txt")


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
def show_split_summary(train_df, test_df, target_col='Quarterly_Total_Claims'):
    """
    Display summary of train-test split.
    """
    print("\n" + "=" * 50)
    print("SPLIT SUMMARY")
    print("=" * 50)

    print(f"\nTrain Set:")
    print(f"  Rows: {len(train_df)}")
    print(f"  Target mean: ${train_df[target_col].mean():,.2f}")
    print(f"  Target std:  ${train_df[target_col].std():,.2f}")

    print(f"\nTest Set:")
    print(f"  Rows: {len(test_df)}")
    print(f"  Target mean: ${test_df[target_col].mean():,.2f}")
    print(f"  Target std:  ${test_df[target_col].std():,.2f}")

    print("=" * 50)


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------
def split_pipeline(input_path="quarterly_claims.csv",
                   split_type="80_20",
                   target_col='Quarterly_Total_Claims'):
    """
    Full split pipeline. Time-based split with 3 preset options.

    Args:
        input_path: path to quarterly_claims.csv
        split_type: '70_30', '80_20', or '90_10'
        target_col: name of target column

    Returns:
        train_df, test_df, X_train, y_train, X_test, y_test
    """
    print("=" * 50)
    print("STARTING TRAIN-TEST SPLIT")
    print("=" * 50)

    df = load_quarterly_data(input_path)
    feature_cols = get_feature_columns(df, target_col=target_col)

    # Select split
    if split_type == "70_30":
        train_df, test_df = split_70_30(df)
        train_path = "train_data_70.csv"
        test_path = "test_data_30.csv"
    elif split_type == "80_20":
        train_df, test_df = split_80_20(df)
        train_path = "train_data_80.csv"
        test_path = "test_data_20.csv"
    elif split_type == "90_10":
        train_df, test_df = split_90_10(df)
        train_path = "train_data_90.csv"
        test_path = "test_data_10.csv"
    else:
        raise ValueError(f"split_type must be '70_30', '80_20', or '90_10'. Got: {split_type}")

    # Extract X, y
    X_train, y_train = extract_xy(train_df, feature_cols, target_col)
    X_test, y_test = extract_xy(test_df, feature_cols, target_col)

    # Save
    save_splits(train_df, test_df, train_path=train_path, test_path=test_path, feature_cols=feature_cols)

    # Summary
    show_split_summary(train_df, test_df, target_col)

    print("\nSplit pipeline complete.")
    return train_df, test_df, X_train, y_train, X_test, y_test


# ---------------------------------------------------------
# RUN ALL THREE SPLITS
# ---------------------------------------------------------
def run_all_splits(input_path="quarterly_claims.csv", target_col='Quarterly_Total_Claims'):
    """
    Run all three split versions and return results.
    """
    results = {}
    for split_type in ["70_30", "80_20", "90_10"]:
        print("\n" + "=" * 60)
        print(f"RUNNING SPLIT: {split_type}")
        print("=" * 60)
        results[split_type] = split_pipeline(input_path, split_type, target_col)
    return results


# ---------------------------------------------------------
# RUN DIRECTLY — ALL THREE SPLITS
# ---------------------------------------------------------
if __name__ == "__main__":
    # Run all three splits by default
    all_results = run_all_splits(
    input_path=r"D:\Insurance Project\outputs\quarterly_claims.csv"
)
    # Access individual results:
    # train_70, test_70, X_train_70, y_train_70, X_test_30, y_test_30 = all_results["70_30"]
    # train_80, test_80, X_train_80, y_train_80, X_test_20, y_test_20 = all_results["80_20"]
    # train_90, test_90, X_train_90, y_train_90, X_test_10, y_test_10 = all_results["90_10"]
