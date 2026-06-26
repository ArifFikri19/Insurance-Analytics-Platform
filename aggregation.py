# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 3: Aggregation — Create Quarterly-Level Dataset
# Uses featured dataset from feature_engineering.py
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# LOAD FEATURED DATA
# ---------------------------------------------------------
def load_featured_data(file_path="featured_insurance_data.csv"):
    """
    Load feature-engineered dataset from step 2.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Featured dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    if 'Incident Date' in df.columns:
        df['Incident Date'] = pd.to_datetime(df['Incident Date'], errors='coerce')

    print("Featured dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    return df


# ---------------------------------------------------------
# QUARTERLY AGGREGATION
# ---------------------------------------------------------
def aggregate_quarterly_claims(df, claim_col='Total Claim Amount'):
    """
    Aggregate individual claims into quarterly summaries.
    This creates the modeling target: Quarterly_Total_Claims.
    """
    if 'Year' not in df.columns or 'Quarter' not in df.columns:
        raise ValueError("'Year' and 'Quarter' columns required. Run preprocessing first.")

    if claim_col not in df.columns:
        raise ValueError(f"Claim column '{claim_col}' not found.")

    # Group by Year and Quarter
    quarterly = df.groupby(['Year', 'Quarter']).agg(
        Quarterly_Total_Claims=(claim_col, 'sum'),
        Quarterly_Claim_Count=(claim_col, 'count'),
        Quarterly_Avg_Claim=(claim_col, 'mean'),
        Quarterly_Median_Claim=(claim_col, 'median'),
        Quarterly_Max_Claim=(claim_col, 'max'),
        Quarterly_Min_Claim=(claim_col, 'min'),
        Quarterly_Std_Claim=(claim_col, 'std'),
        # Customer aggregates
        Unique_Customers=('Customer ID', 'nunique') if 'Customer ID' in df.columns else (claim_col, lambda x: np.nan),
        Avg_Customer_Age=('Age', 'mean') if 'Age' in df.columns else (claim_col, lambda x: np.nan),
        Avg_Months_As_Customer=('Months As Customer', 'mean') if 'Months As Customer' in df.columns else (claim_col, lambda x: np.nan),
        # Policy aggregates
        Avg_Policy_Premium=('Policy Annual Premium', 'mean') if 'Policy Annual Premium' in df.columns else (claim_col, lambda x: np.nan),
        Avg_Policy_Deductable=('Policy Deductable', 'mean') if 'Policy Deductable' in df.columns else (claim_col, lambda x: np.nan),
        # Incident aggregates
        Severe_Incident_Count=('Severe_Incident_Flag', 'sum') if 'Severe_Incident_Flag' in df.columns else (claim_col, lambda x: np.nan),
        High_Deductible_Count=('High_Deductible_Flag', 'sum') if 'High_Deductible_Flag' in df.columns else (claim_col, lambda x: np.nan),
        # Ratio aggregates
        Avg_Injury_Ratio=('Injury_Ratio', 'mean') if 'Injury_Ratio' in df.columns else (claim_col, lambda x: np.nan),
        Avg_Property_Ratio=('Property_Ratio', 'mean') if 'Property_Ratio' in df.columns else (claim_col, lambda x: np.nan),
        Avg_Vehicle_Ratio=('Vehicle_Ratio', 'mean') if 'Vehicle_Ratio' in df.columns else (claim_col, lambda x: np.nan),
    ).reset_index()

    # Create Quarter_Label for readability
    quarterly['Quarter_Label'] = quarterly['Year'].astype(str) + "-Q" + quarterly['Quarter'].astype(str)

    # Sort chronologically
    quarterly = quarterly.sort_values(['Year', 'Quarter']).reset_index(drop=True)

    # Fill NaN std for quarters with only 1 claim
    if 'Quarterly_Std_Claim' in quarterly.columns:
        quarterly['Quarterly_Std_Claim'] = quarterly['Quarterly_Std_Claim'].fillna(0)

    print(f"Quarterly aggregation complete: {len(quarterly)} quarters")
    print(f"Target range: ${quarterly['Quarterly_Total_Claims'].min():,.2f} - ${quarterly['Quarterly_Total_Claims'].max():,.2f}")

    return quarterly


# ---------------------------------------------------------
# LAG FEATURES (for time-series modeling)
# ---------------------------------------------------------
def create_lag_features(quarterly, target_col='Quarterly_Total_Claims', lags=[1, 2, 3, 4]):
    """
    Create lagged versions of the target and key features.
    Essential for time-series forecasting models.
    """
    quarterly = quarterly.copy()

    # Lag target
    for lag in lags:
        quarterly[f'{target_col}_Lag{lag}'] = quarterly[target_col].shift(lag)

    # Lag claim count
    if 'Quarterly_Claim_Count' in quarterly.columns:
        for lag in lags:
            quarterly[f'Quarterly_Claim_Count_Lag{lag}'] = quarterly['Quarterly_Claim_Count'].shift(lag)

    # Lag average claim
    if 'Quarterly_Avg_Claim' in quarterly.columns:
        for lag in lags:
            quarterly[f'Quarterly_Avg_Claim_Lag{lag}'] = quarterly['Quarterly_Avg_Claim'].shift(lag)

    # Rolling averages
    quarterly[f'{target_col}_Rolling_Mean_4Q'] = quarterly[target_col].shift(1).rolling(window=4, min_periods=1).mean()
    quarterly[f'{target_col}_Rolling_Std_4Q'] = quarterly[target_col].shift(1).rolling(window=4, min_periods=1).std().fillna(0)

    print(f"Lag features created: lags {lags}, rolling mean/std (4Q)")
    return quarterly


# ---------------------------------------------------------
# GROWTH FEATURES
# ---------------------------------------------------------
def create_growth_features(quarterly, target_col='Quarterly_Total_Claims'):
    """
    Create quarter-over-quarter growth rates.
    """
    quarterly = quarterly.copy()

    # QoQ growth
    quarterly['QoQ_Growth'] = quarterly[target_col].pct_change()

    # YoY growth (same quarter last year)
    quarterly['YoY_Growth'] = quarterly[target_col].pct_change(periods=4)

    # Absolute difference
    quarterly['QoQ_Absolute_Change'] = quarterly[target_col].diff()

    # Fill NaN for first periods
    quarterly['QoQ_Growth'] = quarterly['QoQ_Growth'].fillna(0)
    quarterly['YoY_Growth'] = quarterly['YoY_Growth'].fillna(0)
    quarterly['QoQ_Absolute_Change'] = quarterly['QoQ_Absolute_Change'].fillna(0)

    print("Growth features created: QoQ_Growth, YoY_Growth, QoQ_Absolute_Change")
    return quarterly


# ---------------------------------------------------------
# SEASONAL FEATURES
# ---------------------------------------------------------
def create_seasonal_features(quarterly):
    """
    Create cyclical encoding for quarter to capture seasonality.
    """
    quarterly = quarterly.copy()

    # Cyclical quarter encoding
    quarterly['Quarter_Sin'] = np.sin(2 * np.pi * quarterly['Quarter'] / 4)
    quarterly['Quarter_Cos'] = np.cos(2 * np.pi * quarterly['Quarter'] / 4)

    # Quarter dummies (alternative to cyclical)
    quarter_dummies = pd.get_dummies(quarterly['Quarter'], prefix='Q', drop_first=False)
    quarterly = pd.concat([quarterly, quarter_dummies], axis=1)

    print("Seasonal features created: Quarter_Sin, Quarter_Cos, Q1-Q4 dummies")
    return quarterly


# ---------------------------------------------------------
# TREND FEATURE
# ---------------------------------------------------------
def create_trend_feature(quarterly):
    """
    Create a simple time trend feature (0, 1, 2, ...).
    """
    quarterly = quarterly.copy()
    quarterly['Time_Trend'] = np.arange(len(quarterly))

    print("Trend feature created: Time_Trend")
    return quarterly


# ---------------------------------------------------------
# CLEAN AGGREGATED DATA
# ---------------------------------------------------------
def clean_aggregated_data(quarterly):
    """
    Handle any remaining NaN values after lag/growth creation.
    """
    quarterly = quarterly.copy()

    # Fill lag NaNs with 0 (first few quarters won't have lags)
    lag_cols = [c for c in quarterly.columns if '_Lag' in c]
    for col in lag_cols:
        quarterly[col] = quarterly[col].fillna(0)

    # Fill any remaining numeric NaNs with 0
    numeric_cols = quarterly.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if quarterly[col].isna().any():
            quarterly[col] = quarterly[col].fillna(0)

    print("Aggregated data cleaned")
    return quarterly


# ---------------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------------
def save_quarterly_data(quarterly, output_path="outputs/quarterly_claims.csv"):
    os.makedirs("outputs", exist_ok=True)
    quarterly.to_csv(output_path, index=False)
    print(f"Quarterly dataset saved: {output_path}")


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
def show_summary(quarterly):
    """
    Display summary of quarterly aggregation.
    """
    print("\n" + "=" * 50)
    print("AGGREGATION COMPLETE")
    print("=" * 50)
    print(f"Shape: {quarterly.shape}")
    print(f"Quarters: {len(quarterly)}")
    print(f"Date range: {quarterly['Quarter_Label'].iloc[0]} to {quarterly['Quarter_Label'].iloc[-1]}")
    print(f"\nTarget Statistics:")
    print(f"  Mean: ${quarterly['Quarterly_Total_Claims'].mean():,.2f}")
    print(f"  Std:  ${quarterly['Quarterly_Total_Claims'].std():,.2f}")
    print(f"  Min:  ${quarterly['Quarterly_Total_Claims'].min():,.2f}")
    print(f"  Max:  ${quarterly['Quarterly_Total_Claims'].max():,.2f}")
    print(f"\nColumns: {list(quarterly.columns)}")
    print("\nPreview:")
    print(quarterly.head())
    print("=" * 50)


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------
def aggregation_pipeline(input_path="featured_insurance_data.csv"):
    """
    Full aggregation pipeline.
    Returns quarterly aggregated dataframe ready for modeling.
    """
    print("=" * 50)
    print("STARTING AGGREGATION")
    print("=" * 50)

    df = load_featured_data(input_path)
    quarterly = aggregate_quarterly_claims(df)
    quarterly = create_lag_features(quarterly)
    quarterly = create_growth_features(quarterly)
    quarterly = create_seasonal_features(quarterly)
    quarterly = create_trend_feature(quarterly)
    quarterly = clean_aggregated_data(quarterly)
    save_quarterly_data(quarterly)
    show_summary(quarterly)

    return quarterly


# ---------------------------------------------------------
# RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    quarterly_df = aggregation_pipeline()