# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 5: Train Models + Pre-Compute Analysis Tables
# Trains regression models, generates risk levels,
# and outputs tables for dashboard visualization.
# Includes Naive Baseline: prediction = previous quarter
# Now outputs ALL quarters (train + test) for full dashboard view
# ---------------------------------------------------------

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback if not installed
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed. Only Linear + Random Forest will be trained.")


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TARGET_COL = 'Quarterly_Total_Claims'
RISK_QUANTILES = [0.33, 0.67]  # LOW / MEDIUM / HIGH thresholds
SPLITS = {
    '70_30': ('train_data_70.csv', 'test_data_30.csv'),
    '80_20': ('train_data_80.csv', 'test_data_20.csv'),
    '90_10': ('train_data_90.csv', 'test_data_10.csv'),
}

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
# LOAD DATA
# ---------------------------------------------------------
def load_split(train_path, test_path):
    """Load train/test CSVs and extract X, y."""
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing split files: {train_path} or {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    drop_cols = [TARGET_COL, 'Quarter_Label', 'Year', 'Quarter']
    feature_cols = [c for c in FORECAST_SAFE_FEATURES if c in train_df.columns]
    if not feature_cols:
        feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    return train_df, test_df, X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------
# NAIVE BASELINE (previous quarter = prediction)
# ---------------------------------------------------------
def generate_naive_baseline(y_train, y_test):
    """
    Naive baseline: predict current quarter = previous quarter's actual.
    First test prediction = last training value.
    """
    last_train = y_train.iloc[-1]
    predictions = [last_train]
    predictions.extend(y_test.iloc[:-1].tolist())
    return np.array(predictions)


def generate_naive_baseline_train(y_train):
    """
    Naive baseline for training data: prediction = previous quarter's actual.
    First prediction = same as actual (no previous available).
    """
    predictions = [y_train.iloc[0]]  # First quarter: no previous, use itself
    predictions.extend(y_train.iloc[:-1].tolist())
    return np.array(predictions)


# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------
def train_linear_regression(X_train, y_train):
    """Train Linear Regression with scaling."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    return {'model': model, 'scaler': scaler, 'name': 'LinearRegression'}


def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return {'model': model, 'scaler': None, 'name': 'RandomForest'}


def train_xgboost(X_train, y_train, n_estimators=200, random_state=42):
    """Train XGBoost Regressor."""
    if not HAS_XGB:
        return None
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return {'model': model, 'scaler': None, 'name': 'XGBoost'}


def train_ensemble(X_train, y_train, models_dict):
    """
    Train an Ensemble (VotingRegressor) from all available trained models.
    This creates a proper Ensemble.pkl that can be loaded for prediction.
    """
    estimators = []
    for name, model_dict in models_dict.items():
        if model_dict is not None and model_dict.get('scaler') is None:
            # Only include models that don't need a scaler
            # (VotingRegressor can't handle per-model scaling)
            estimators.append((name, model_dict['model']))

    # If LinearRegression (which uses scaler) is the only model, retrain without scaler
    if not estimators:
        # Fallback: train a simple LR without scaler for ensemble
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        estimators.append(('LinearRegression', lr))

    if len(estimators) < 2:
        # Need at least 2 models for an ensemble to make sense
        # Add a simple LinearRegression as second model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        if 'LinearRegression' not in [e[0] for e in estimators]:
            estimators.append(('LinearRegression', lr))
        else:
            # Already have LR, add a smaller RF
            rf_small = RandomForestRegressor(n_estimators=50, random_state=99, n_jobs=-1)
            rf_small.fit(X_train, y_train)
            estimators.append(('RandomForest_Small', rf_small))

    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(X_train, y_train)

    return {'model': ensemble, 'scaler': None, 'name': 'Ensemble'}


def train_all_models(X_train, y_train):
    """Train all available models including Ensemble."""
    models = {}
    models['LinearRegression'] = train_linear_regression(X_train, y_train)
    models['RandomForest'] = train_random_forest(X_train, y_train)
    if HAS_XGB:
        models['XGBoost'] = train_xgboost(X_train, y_train)

    # Train Ensemble from the individual models
    models['Ensemble'] = train_ensemble(X_train, y_train, models)

    return models


# ---------------------------------------------------------
# PREDICT & EVALUATE
# ---------------------------------------------------------
def predict_model(model_dict, X):
    """Run prediction using model dict (may include scaler)."""
    model = model_dict['model']
    scaler = model_dict.get('scaler')
    if scaler:
        X = scaler.transform(X)
    return model.predict(X)


def evaluate_model(y_true, y_pred, model_name, split_name):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    results = {
        'split': split_name,
        'model': model_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE_%': round(mape, 2)
    }
    return results


# ---------------------------------------------------------
# RISK LEVELS
# ---------------------------------------------------------
def compute_risk_thresholds(train_y):
    """
    Compute LOW / MEDIUM / HIGH thresholds from training data.
    """
    q33 = train_y.quantile(RISK_QUANTILES[0])
    q67 = train_y.quantile(RISK_QUANTILES[1])
    thresholds = {
        'low_max': q33,
        'medium_max': q67,
        'labels': ['LOW', 'MEDIUM', 'HIGH']
    }
    return thresholds


def assign_risk_level(prediction, thresholds):
    """Assign LOW/MEDIUM/HIGH based on thresholds."""
    if prediction <= thresholds['low_max']:
        return 'LOW'
    elif prediction <= thresholds['medium_max']:
        return 'MEDIUM'
    else:
        return 'HIGH'


# ---------------------------------------------------------
# PRE-COMPUTE DASHBOARD TABLES
# ---------------------------------------------------------
def compute_customer_risk_scores(featured_path="featured_insurance_data.csv"):
    """
    Pre-compute customer-level risk scores from row-level data.
    Uses: Age, incident type, claim frequency, total claim amount.
    """
    if not os.path.exists(featured_path):
        print(f"Warning: {featured_path} not found. Skipping customer risk scores.")
        return None

    df = pd.read_csv(featured_path)

    # Aggregate per customer
    if 'Customer ID' not in df.columns:
        print("Warning: 'Customer ID' not found. Skipping customer risk scores.")
        return None

    customer = df.groupby('Customer ID').agg(
        Age=('Age', 'first'),
        Incident_Count=('Customer ID', 'count'),
        Total_Claim=('Total Claim Amount', 'sum'),
        Avg_Claim=('Total Claim Amount', 'mean'),
        Severe_Count=('Severe_Incident_Flag', 'sum') if 'Severe_Incident_Flag' in df.columns else ('Total Claim Amount', lambda x: 0),
        Top_Incident_Type=('Incident Type', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown') if 'Incident Type' in df.columns else ('Total Claim Amount', lambda x: 'Unknown')
    ).reset_index()

    # Simple risk score: normalize and weight
    customer['Claim_Frequency_Score'] = (customer['Incident_Count'] - customer['Incident_Count'].min()) / (customer['Incident_Count'].max() - customer['Incident_Count'].min() + 1e-8)
    customer['Claim_Amount_Score'] = (customer['Total_Claim'] - customer['Total_Claim'].min()) / (customer['Total_Claim'].max() - customer['Total_Claim'].min() + 1e-8)

    if 'Severe_Count' in customer.columns:
        customer['Severity_Score'] = (customer['Severe_Count'] - customer['Severe_Count'].min()) / (customer['Severe_Count'].max() - customer['Severe_Count'].min() + 1e-8)
    else:
        customer['Severity_Score'] = 0

    # Composite risk score (0-100)
    customer['Risk_Score'] = (
        customer['Claim_Frequency_Score'] * 30 +
        customer['Claim_Amount_Score'] * 40 +
        customer['Severity_Score'] * 30
    ).round(1)

    # Risk category
    customer['Risk_Category'] = pd.cut(
        customer['Risk_Score'],
        bins=[0, 30, 60, 100],
        labels=['LOW', 'MEDIUM', 'HIGH'],
        include_lowest=True
    )

    customer = customer.sort_values('Risk_Score', ascending=False)
    return customer


def compute_quarterly_incident_patterns(featured_path="featured_insurance_data.csv"):
    """
    Pre-compute incident type counts and costs per quarter.
    """
    if not os.path.exists(featured_path):
        return None

    df = pd.read_csv(featured_path)

    if 'Incident Type' not in df.columns or 'Year' not in df.columns:
        return None

    # Quarterly aggregation by incident type
    patterns = df.groupby(['Year', 'Quarter', 'Incident Type']).agg(
        Count=('Incident Type', 'count'),
        Total_Cost=('Total Claim Amount', 'sum'),
        Avg_Cost=('Total Claim Amount', 'mean')
    ).reset_index()

    patterns['Quarter_Label'] = patterns['Year'].astype(str) + '-Q' + patterns['Quarter'].astype(str)
    patterns = patterns.sort_values(['Year', 'Quarter', 'Total_Cost'], ascending=[True, True, False])

    return patterns


def compute_quarterly_claim_breakdown(featured_path="featured_insurance_data.csv"):
    """
    Pre-compute Injury / Property / Vehicle claim breakdown per quarter.
    """
    if not os.path.exists(featured_path):
        return None

    df = pd.read_csv(featured_path)

    if 'Year' not in df.columns:
        return None

    agg_dict = {}
    for col in ['Injury Claim', 'Property Claim', 'Vehicle Claim', 'Total Claim Amount']:
        if col in df.columns:
            agg_dict[col] = 'sum'

    breakdown = df.groupby(['Year', 'Quarter']).agg(agg_dict).reset_index()
    breakdown['Quarter_Label'] = breakdown['Year'].astype(str) + '-Q' + breakdown['Quarter'].astype(str)

    # Calculate contribution percentages
    if 'Total Claim Amount' in breakdown.columns:
        for col in ['Injury Claim', 'Property Claim', 'Vehicle Claim']:
            if col in breakdown.columns:
                pct_col = col.replace('Claim', 'Pct')
                breakdown[pct_col] = (breakdown[col] / breakdown['Total Claim Amount'] * 100).round(2)

    return breakdown


# ---------------------------------------------------------
# SAVE MODELS & OUTPUTS
# ---------------------------------------------------------
def save_model_artifacts(models, thresholds, split_name, feature_cols):
    """Save trained models and thresholds."""
    os.makedirs(f"models/{split_name}", exist_ok=True)

    for name, model_dict in models.items():
        path = f"models/{split_name}/{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"  Saved: {path}")

    # Save thresholds
    with open(f"models/{split_name}/risk_thresholds.json", 'w') as f:
        json.dump({
            'low_max': float(thresholds['low_max']),
            'medium_max': float(thresholds['medium_max'])
        }, f)

    # Save feature columns
    with open(f"models/{split_name}/feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_cols))


def save_predictions(train_df, test_df, models, predictions_dict, thresholds,
                     split_name, feature_cols, naive_pred=None):
    """
    Save predictions for ALL quarters (train + test) for full dashboard view.
    Includes a 'Data_Split' column to distinguish train vs test predictions.
    """
    os.makedirs("outputs", exist_ok=True)

    # =========================================================
    # PART 1: TEST predictions (out-of-sample — the real evaluation)
    # =========================================================
    test_result = test_df[['Year', 'Quarter', 'Quarter_Label', TARGET_COL]].copy()
    test_result.rename(columns={TARGET_COL: 'Actual'}, inplace=True)
    test_result['Data_Split'] = 'Test'

    # Add Naive Baseline for test
    if naive_pred is not None:
        test_result['Naive_Baseline_Prediction'] = naive_pred
        test_result['Naive_Baseline_Risk'] = [
            assign_risk_level(p, thresholds) for p in naive_pred
        ]

    # Add ML model predictions for test
    for model_name, preds in predictions_dict.items():
        test_result[f'{model_name}_Prediction'] = preds
        test_result[f'{model_name}_Risk'] = [
            assign_risk_level(p, thresholds) for p in preds
        ]

    # =========================================================
    # PART 2: TRAIN predictions (in-sample — for full timeline view)
    # =========================================================
    train_result = train_df[['Year', 'Quarter', 'Quarter_Label', TARGET_COL]].copy()
    train_result.rename(columns={TARGET_COL: 'Actual'}, inplace=True)
    train_result['Data_Split'] = 'Train'

    # Naive baseline for training data
    y_train = train_df[TARGET_COL]
    naive_train = generate_naive_baseline_train(y_train)
    train_result['Naive_Baseline_Prediction'] = naive_train
    train_result['Naive_Baseline_Risk'] = [
        assign_risk_level(p, thresholds) for p in naive_train
    ]

    # Generate predictions for training data using each model
    X_train = train_df[feature_cols]
    for model_name, model_dict in models.items():
        train_preds = predict_model(model_dict, X_train)
        train_result[f'{model_name}_Prediction'] = train_preds
        train_result[f'{model_name}_Risk'] = [
            assign_risk_level(p, thresholds) for p in train_preds
        ]

    # =========================================================
    # COMBINE: Train + Test (full timeline)
    # =========================================================
    result_df = pd.concat([train_result, test_result], ignore_index=True)
    result_df = result_df.sort_values(['Year', 'Quarter']).reset_index(drop=True)

    path = f"outputs/predictions_{split_name}.csv"
    result_df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(train_result)} train + {len(test_result)} test = {len(result_df)} total quarters)")

    return result_df


def save_metrics(all_metrics, split_name):
    """Save evaluation metrics."""
    os.makedirs("outputs", exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    path = f"outputs/metrics_{split_name}.csv"
    metrics_df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return metrics_df


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
def show_training_summary(all_results):
    """Display training summary across all splits."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for split_name, data in all_results.items():
        print(f"\nSplit: {split_name}")
        print(f"  Train rows: {data['train_shape']}")
        print(f"  Test rows:  {data['test_shape']}")
        print(f"  Risk thresholds: LOW ≤ ${data['thresholds']['low_max']:,.0f}, MEDIUM ≤ ${data['thresholds']['medium_max']:,.0f}")
        print("  Model Performance (Test Set Only):")
        for m in data['metrics']:
            print(f"    {m['model']:15s} | MAE: ${m['MAE']:>10,.2f} | RMSE: ${m['RMSE']:>10,.2f} | R²: {m['R2']:.4f} | MAPE: {m['MAPE_%']:.2f}%")

    print("\n" + "=" * 60)
    print("Output files:")
    print("  outputs/predictions_70_30.csv  (ALL quarters: train + test)")
    print("  outputs/predictions_80_20.csv  (ALL quarters: train + test)")
    print("  outputs/predictions_90_10.csv  (ALL quarters: train + test)")
    print("  outputs/metrics_70_30.csv      (test-set metrics only)")
    print("  outputs/metrics_80_20.csv      (test-set metrics only)")
    print("  outputs/metrics_90_10.csv      (test-set metrics only)")
    print("  outputs/customer_risk_scores.csv")
    print("  outputs/quarterly_incident_patterns.csv")
    print("  outputs/quarterly_claim_breakdown.csv")
    print("\n  models/70_30/Ensemble.pkl")
    print("  models/80_20/Ensemble.pkl")
    print("  models/90_10/Ensemble.pkl")
    print("\n  Column 'Data_Split' in predictions files:")
    print("    'Train' = in-sample (model saw this data)")
    print("    'Test'  = out-of-sample (true predictive performance)")
    print("\n  Naive Baseline: prediction = previous quarter's actual")
    print("  Ensemble: VotingRegressor (average of all ML models)")
    print("  If ML models cannot beat Naive on TEST data, features may be insufficient.")
    print("=" * 60)


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------
def train_models_pipeline():
    """
    Full training pipeline.
    Trains models on all three splits, pre-computes dashboard tables.
    Outputs predictions for ALL quarters (train + test) for full dashboard view.
    """
    print("=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)

    all_results = {}

    # Train on each split
    for split_name, (train_path, test_path) in SPLITS.items():
        print(f"\n--- Processing Split: {split_name} ---")

        # Load
        train_df, test_df, X_train, y_train, X_test, y_test, feature_cols = load_split(train_path, test_path)

        # Compute risk thresholds from training data
        thresholds = compute_risk_thresholds(y_train)
        print(f"Risk thresholds: LOW ≤ ${thresholds['low_max']:,.0f}, MEDIUM ≤ ${thresholds['medium_max']:,.0f}")

        # Train models (now includes Ensemble)
        models = train_all_models(X_train, y_train)

        # Predict & evaluate on TEST set
        predictions_dict = {}
        all_metrics = []

        # Naive baseline (previous quarter = prediction) — TEST only
        naive_pred = generate_naive_baseline(y_train, y_test)
        naive_metrics = evaluate_model(y_test, naive_pred, 'Naive_Baseline', split_name)
        all_metrics.append(naive_metrics)
        print(f"  {'Naive_Baseline':15s} | MAE: ${naive_metrics['MAE']:>10,.2f} | RMSE: ${naive_metrics['RMSE']:>10,.2f} | R²: {naive_metrics['R2']:.4f} (previous quarter = prediction)")

        for name, model_dict in models.items():
            y_pred = predict_model(model_dict, X_test)
            predictions_dict[name] = y_pred
            metrics = evaluate_model(y_test, y_pred, name, split_name)
            all_metrics.append(metrics)
            print(f"  {name:15s} | MAE: ${metrics['MAE']:>10,.2f} | RMSE: ${metrics['RMSE']:>10,.2f} | R²: {metrics['R2']:.4f}")

        # Check if best ML model beats naive baseline
        ml_metrics = [m for m in all_metrics if m['model'] != 'Naive_Baseline']
        if ml_metrics:
            best_ml = min(ml_metrics, key=lambda x: x['RMSE'])
            baseline_beaten = best_ml['RMSE'] < naive_metrics['RMSE']
            print(f"\n  {'='*54}")
            if baseline_beaten:
                print(f"  ✅ Best ML model ({best_ml['model']}) BEATS naive baseline")
                print(f"     ML RMSE: ${best_ml['RMSE']:,.2f} < Naive RMSE: ${naive_metrics['RMSE']:,.2f}")
            else:
                print(f"  ⚠️  WARNING: No ML model beats naive baseline!")
                print(f"     Best ML RMSE: ${best_ml['RMSE']:,.2f} >= Naive RMSE: ${naive_metrics['RMSE']:,.2f}")
                print(f"     Consider more features or different model architecture.")
            print(f"  {'='*54}")

        # Save artifacts (includes Ensemble.pkl)
        save_model_artifacts(models, thresholds, split_name, feature_cols)

        # Save predictions for ALL quarters (train + test)
        save_predictions(
            train_df, test_df, models, predictions_dict, thresholds,
            split_name, feature_cols, naive_pred=naive_pred
        )

        # Save metrics (test-set only — real evaluation)
        save_metrics(all_metrics, split_name)

        all_results[split_name] = {
            'train_shape': len(train_df),
            'test_shape': len(test_df),
            'thresholds': thresholds,
            'metrics': all_metrics,
            'models': list(models.keys())
        }

    # Pre-compute dashboard tables (row-level analysis, split-agnostic)
    print("\n--- Pre-computing Dashboard Tables ---")

    customer_risk = compute_customer_risk_scores()
    if customer_risk is not None:
        customer_risk.to_csv("outputs/customer_risk_scores.csv", index=False)
        print("  Saved: outputs/customer_risk_scores.csv")

    incident_patterns = compute_quarterly_incident_patterns()
    if incident_patterns is not None:
        incident_patterns.to_csv("outputs/quarterly_incident_patterns.csv", index=False)
        print("  Saved: outputs/quarterly_incident_patterns.csv")

    claim_breakdown = compute_quarterly_claim_breakdown()
    if claim_breakdown is not None:
        claim_breakdown.to_csv("outputs/quarterly_claim_breakdown.csv", index=False)
        print("  Saved: outputs/quarterly_claim_breakdown.csv")

    # Summary
    show_training_summary(all_results)

    return all_results


# ---------------------------------------------------------
# RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    results = train_models_pipeline()
