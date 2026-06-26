# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 2: Feature Engineering (Production Ready)
# Uses cleaned dataset from preprocessing.py
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# LOAD CLEAN DATA
# ---------------------------------------------------------
def load_clean_data(file_path="cleaned_insurance_data.csv"):
    """
    Load cleaned dataset from preprocessing.py.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    if 'Incident Date' in df.columns:
        df['Incident Date'] = pd.to_datetime(df['Incident Date'], errors='coerce')

    print("Clean dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    return df


# ---------------------------------------------------------
# TIME FEATURES
# ---------------------------------------------------------
def create_time_features(df):
    """
    Create additional time features NOT created in preprocessing.py.
    """
    if 'Incident Date' not in df.columns:
        print("Warning: 'Incident Date' not found. Skipping time features.")
        return df

    df['DayOfWeek'] = df['Incident Date'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Quarter_Label'] = df['Year'].astype(str) + "-Q" + df['Quarter'].astype(str)

    if 'Policy Start Date' in df.columns:
        df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
        df['Days_Since_Policy_Start'] = (
            df['Incident Date'] - df['Policy Start Date']
        ).dt.days

    print("Time features created: DayOfWeek, Is_Weekend, Quarter_Label")
    return df


# ---------------------------------------------------------
# CLAIM RATIO FEATURES
# ---------------------------------------------------------
def create_claim_ratio_features(df):
    """
    Create claim component ratios.
    """
    if 'Total Claim Amount' not in df.columns:
        print("Warning: 'Total Claim Amount' not found. Skipping claim ratios.")
        return df

    df['Has_Positive_Claim'] = (df['Total Claim Amount'] > 0).astype(int)

    total = df['Total Claim Amount'].replace(0, np.nan)

    if 'Injury Claim' in df.columns:
        df['Injury_Ratio'] = (df['Injury Claim'] / total).fillna(0)

    if 'Property Claim' in df.columns:
        df['Property_Ratio'] = (df['Property Claim'] / total).fillna(0)

    if 'Vehicle Claim' in df.columns:
        df['Vehicle_Ratio'] = (df['Vehicle Claim'] / total).fillna(0)

    print("Claim ratio features created.")
    return df


# ---------------------------------------------------------
# CUSTOMER FEATURES
# ---------------------------------------------------------
def create_customer_features(df):
    """
    Customer-related features.
    """
    if 'Total Claim Amount' in df.columns and 'Months As Customer' in df.columns:
        months = df['Months As Customer'].replace(0, np.nan)
        df['Claim_per_Month'] = (df['Total Claim Amount'] / months).fillna(0)
        df['Months_Is_Zero'] = (df['Months As Customer'] == 0).astype(int)

    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 25, 40, 60, 120],
            labels=['18-25', '26-40', '41-60', '60+'],
            include_lowest=True
        )

    print("Customer features created.")
    return df


# ---------------------------------------------------------
# POLICY FEATURES
# ---------------------------------------------------------
def create_policy_features(df):
    """
    Policy-related features.
    """
    if 'Policy Annual Premium' in df.columns and 'Total Claim Amount' in df.columns:
        premium = df['Policy Annual Premium'].replace(0, np.nan)
        df['Claim_to_Premium_Ratio'] = (
            df['Total Claim Amount'] / premium
        ).fillna(0)

        df['Premium_Is_Zero'] = (
            df['Policy Annual Premium'] == 0
        ).astype(int)

    if 'Policy Deductable' in df.columns:
        df['High_Deductible_Flag'] = (
            df['Policy Deductable'] >= 1500
        ).astype(int)

    print("Policy features created.")
    return df


# ---------------------------------------------------------
# INCIDENT FEATURES
# ---------------------------------------------------------
def create_incident_features(df):
    """
    Incident severity and type features.
    """
    if 'Incident Severity' in df.columns:
        severe_labels = ['Major Damage', 'Total Loss', 'Severe']

        df['Severe_Incident_Flag'] = (
            df['Incident Severity'].isin(severe_labels)
        ).astype(int)

    print("Incident features created.")
    return df


# ---------------------------------------------------------
# CLEAN FEATURE VALUES
# ---------------------------------------------------------
def clean_feature_values(df):
    """
    Replace inf with NaN, then fill engineered values.
    """
    df = df.replace([np.inf, -np.inf], np.nan)

    engineered_numeric = [
        'Injury_Ratio',
        'Property_Ratio',
        'Vehicle_Ratio',
        'Claim_per_Month',
        'Claim_to_Premium_Ratio',
        'Days_Since_Policy_Start'
    ]

    for col in engineered_numeric:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    print("Feature values cleaned.")
    return df


# ---------------------------------------------------------
# NEW OUTPUT: CUSTOMER RISK SCORES
# ---------------------------------------------------------
def create_customer_risk_scores(df):
    """
    Create customer risk score output for dashboard.
    """
    if 'Customer ID' not in df.columns:
        print("Customer ID not found. Skipping risk scores.")
        return

    risk = df.groupby("Customer ID").agg(
        Total_Claim=("Total Claim Amount", "sum"),
        Incident_Count=("Customer ID", "count")
    ).reset_index()

    risk["Risk_Score"] = (
        risk["Total_Claim"] / 1000 +
        risk["Incident_Count"] * 5
    )

    risk["Risk_Category"] = pd.cut(
        risk["Risk_Score"],
        bins=[-1, 50, 150, 999999999],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    risk["Top_Incident_Type"] = "Mixed"

    os.makedirs("outputs", exist_ok=True)
    risk.to_csv("outputs/customer_risk_scores.csv", index=False)

    print("Saved: outputs/customer_risk_scores.csv")


# ---------------------------------------------------------
# NEW OUTPUT: CLAIM BREAKDOWN
# ---------------------------------------------------------
def create_claim_breakdown(df):
    """
    Create quarterly claim breakdown output.
    """
    needed = ['Year', 'Quarter']

    if not all(col in df.columns for col in needed):
        print("Year/Quarter missing. Skipping claim breakdown.")
        return

    injury = 'Injury Claim' if 'Injury Claim' in df.columns else None
    prop = 'Property Claim' if 'Property Claim' in df.columns else None
    vehicle = 'Vehicle Claim' if 'Vehicle Claim' in df.columns else None

    agg_dict = {}

    if injury:
        agg_dict[injury] = 'sum'
    if prop:
        agg_dict[prop] = 'sum'
    if vehicle:
        agg_dict[vehicle] = 'sum'

    if not agg_dict:
        print("No claim columns found.")
        return

    summary = df.groupby(['Year', 'Quarter']).agg(agg_dict).reset_index()

    summary["Quarter_Label"] = (
        summary["Year"].astype(str) +
        "-Q" +
        summary["Quarter"].astype(str)
    )

    total = 0

    if injury:
        total += summary[injury]
    if prop:
        total += summary[prop]
    if vehicle:
        total += summary[vehicle]

    total = total.replace(0, np.nan)

    if injury:
        summary["Injury Pct"] = (summary[injury] / total * 100).fillna(0)

    if prop:
        summary["Property Pct"] = (summary[prop] / total * 100).fillna(0)

    if vehicle:
        summary["Vehicle Pct"] = (summary[vehicle] / total * 100).fillna(0)

    os.makedirs("outputs", exist_ok=True)
    summary.to_csv("outputs/quarterly_claim_breakdown.csv", index=False)

    print("Saved: outputs/quarterly_claim_breakdown.csv")


# ---------------------------------------------------------
# NEW OUTPUT: INCIDENT PATTERNS
# ---------------------------------------------------------
def create_incident_patterns(df):
    """
    Create quarterly incident pattern output.
    """
    needed = ['Year', 'Quarter', 'Incident Type']

    if not all(col in df.columns for col in needed):
        print("Incident columns missing. Skipping patterns.")
        return

    patt = df.groupby(
        ['Year', 'Quarter', 'Incident Type']
    ).agg(
        Count=('Incident Type', 'count'),
        Total_Cost=('Total Claim Amount', 'sum')
    ).reset_index()

    patt["Quarter_Label"] = (
        patt["Year"].astype(str) +
        "-Q" +
        patt["Quarter"].astype(str)
    )

    os.makedirs("outputs", exist_ok=True)
    patt.to_csv("outputs/quarterly_incident_patterns.csv", index=False)

    print("Saved: outputs/quarterly_incident_patterns.csv")


# ---------------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------------
def save_engineered_data(df, output_path="featured_insurance_data.csv"):
    """
    Save feature engineered dataset.
    """
    df.to_csv(output_path, index=False)
    print(f"Feature engineered dataset saved: {output_path}")


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
def show_summary(df):
    """
    Display summary of engineered features.
    """
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 50)
    print(f"Final Shape: {df.shape}")
    print(df.head())
    print("=" * 50)


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------
def feature_engineering_pipeline(input_path="cleaned_insurance_data.csv"):
    """
    Full feature engineering pipeline.
    Returns engineered dataframe.
    """
    print("=" * 50)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 50)

    df = load_clean_data(input_path)
    df = create_time_features(df)
    df = create_claim_ratio_features(df)
    df = create_customer_features(df)
    df = create_policy_features(df)
    df = create_incident_features(df)
    df = clean_feature_values(df)

    # ORIGINAL SAVE
    save_engineered_data(df)

    # NEW OUTPUTS FOR DASHBOARD
    create_customer_risk_scores(df)
    create_claim_breakdown(df)
    create_incident_patterns(df)

    show_summary(df)

    return df


# ---------------------------------------------------------
# RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    df = feature_engineering_pipeline()