# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 6: Hyperparameter Tuning (Time-Series Aware)
# Uses train/test splits from split_data.py
# ---------------------------------------------------------

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Suppress sklearn/joblib nested parallelism warning (harmless, cosmetic only)
warnings.filterwarnings('ignore', message=r'.*sklearn\.utils\.parallel\.delayed.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')

# XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TARGET_COL = 'Quarterly_Total_Claims'
N_SPLITS_CV = 2      # Safer for small quarterly datasets
N_ITER = 20          # RandomizedSearch iterations
RANDOM_STATE = 42

# EXPLICIT train/test pairs (never rely on string replacement)
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
# LOAD TRAINING DATA
# ---------------------------------------------------------
def load_train_data(train_path, test_path):
    """Load training and test CSVs, extract X, y, and feature columns."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    drop_cols = [TARGET_COL, 'Quarter_Label', 'Year', 'Quarter']

    # Derive forecast-safe features ONCE from training data
    feature_cols = [c for c in FORECAST_SAFE_FEATURES if c in train_df.columns]
    if not feature_cols:
        feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    # Use SAME feature columns for test (safety guard)
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test data missing features: {missing}")

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    # Safety: assert alignment
    assert list(X_train.columns) == list(X_test.columns), \
        "Train and test feature columns do not match!"

    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------
# TIME-SERIES CROSS-VALIDATION
# ---------------------------------------------------------
def get_ts_cv():
    """Return TimeSeriesSplit for temporal-aware CV."""
    return TimeSeriesSplit(n_splits=N_SPLITS_CV)


def has_enough_cv_samples(X, split_name):
    """TimeSeriesSplit requires more samples than folds."""
    min_samples = N_SPLITS_CV + 1
    if len(X) < min_samples:
        print(
            f"  Skipping tuning for {split_name}: "
            f"{len(X)} training samples is too small for TimeSeriesSplit({N_SPLITS_CV}). "
            f"Need at least {min_samples}."
        )
        return False
    return True


# ---------------------------------------------------------
# TUNE RANDOM FOREST
# ---------------------------------------------------------
def tune_random_forest(X, y):
    """
    Tune Random Forest using RandomizedSearchCV + TimeSeriesSplit.
    """
    print("\n  Tuning Random Forest...")

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    tscv = get_ts_cv()

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X, y)

    print(f"    Best RMSE (CV): ${-search.best_score_:,.2f}")
    print(f"    Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, -search.best_score_


# ---------------------------------------------------------
# TUNE XGBOOST
# ---------------------------------------------------------
def tune_xgboost(X, y):
    """
    Tune XGBoost using RandomizedSearchCV + TimeSeriesSplit.
    """
    if not HAS_XGB:
        print("  XGBoost not installed → skipping tuning")
        return None, None, None

    print("\n  Tuning XGBoost...")

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 5, 10]
    }

    model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    tscv = get_ts_cv()

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X, y)

    print(f"    Best RMSE (CV): ${-search.best_score_:,.2f}")
    print(f"    Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, -search.best_score_



# ---------------------------------------------------------
# EVALUATE TUNED MODELS ON TEST SET
# ---------------------------------------------------------
def evaluate_on_test(model, X_test, y_test, model_name, split_name):
    """Evaluate tuned model on held-out test set."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

    # Strip '_Tuned' suffix for comparison matching
    base_name = model_name.replace('_Tuned', '')

    results = {
        'split': split_name,
        'model': base_name,  # use base name so comparison matches untuned
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE_%': round(mape, 2)
    }
    return results, y_pred


# ---------------------------------------------------------
# SAVE TUNED MODELS
# ---------------------------------------------------------
def save_tuned_model(model, params, cv_rmse, model_name, split_name, feature_cols):
    """Save tuned model, params, and metadata."""
    dir_path = f"models_tuned/{split_name}"
    os.makedirs(dir_path, exist_ok=True)

    # Save model (dict wrapper for consistency with predict_model API)
    model_path = f"{dir_path}/{model_name}_tuned.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': None, 'name': model_name}, f)

    # Save params + CV score
    meta = {
        'model_name': model_name,
        'split': split_name,
        'best_params': params,
        'cv_rmse': float(cv_rmse)
    }
    meta_path = f"{dir_path}/{model_name}_tuned_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Save feature columns
    with open(f"{dir_path}/feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"    Saved: {model_path}")
    print(f"    Saved: {meta_path}")


# ---------------------------------------------------------
# SAVE COMPARISON TABLE
# ---------------------------------------------------------
def save_comparison(metrics_before, metrics_after, split_name):
    """Save before/after tuning comparison."""
    os.makedirs("outputs", exist_ok=True)

    comparison = []
    for after in metrics_after:
        # Find matching 'before' metric
        before = next(
            (b for b in metrics_before if b['model'] == after['model']),
            None
        )
        if before:
            comparison.append({
                'model': after['model'],
                'split': split_name,
                'RMSE_before': before['RMSE'],
                'RMSE_after': after['RMSE'],
                'RMSE_improvement_%': round((before['RMSE'] - after['RMSE']) / before['RMSE'] * 100, 2),
                'MAE_before': before['MAE'],
                'MAE_after': after['MAE'],
                'R2_before': before['R2'],
                'R2_after': after['R2'],
            })

    df = pd.DataFrame(comparison)
    path = f"outputs/tuning_comparison_{split_name}.csv"
    df.to_csv(path, index=False)
    print(f"\n  Tuning comparison saved: {path}")
    return df


# ---------------------------------------------------------
# LOAD PREVIOUS METRICS
# ---------------------------------------------------------
def load_previous_metrics(split_name):
    """Load metrics from train_models.py for comparison."""
    path = f"outputs/metrics_{split_name}.csv"
    if not os.path.exists(path):
        print(f"  Warning: {path} not found. Cannot compare to untuned models.")
        return []
    df = pd.read_csv(path)
    return df.to_dict('records')


# ---------------------------------------------------------
# RUN TUNING FOR ONE SPLIT
# ---------------------------------------------------------
def tune_split(split_name, train_path, test_path):
    """Run full tuning pipeline for one split."""
    print(f"\n{'='*60}")
    print(f"TUNING SPLIT: {split_name}")
    print(f"{'='*60}")

    # Load data (explicit paths, aligned features)
    X_train, y_train, X_test, y_test, feature_cols = load_train_data(train_path, test_path)

    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Features: {len(feature_cols)}")

    if not has_enough_cv_samples(X_train, split_name):
        return {}, []

    # Load previous metrics for comparison
    previous_metrics = load_previous_metrics(split_name)

    # Tune models
    tuned_results = []
    tuned_models = {}

    # Random Forest
    rf_model, rf_params, rf_cv_rmse = tune_random_forest(X_train, y_train)
    if rf_model is not None:
        save_tuned_model(rf_model, rf_params, rf_cv_rmse, 'RandomForest', split_name, feature_cols)
        metrics, preds = evaluate_on_test(rf_model, X_test, y_test, 'RandomForest_Tuned', split_name)
        tuned_results.append(metrics)
        tuned_models['RandomForest_Tuned'] = {'model': rf_model, 'preds': preds, 'params': rf_params}
        print(f"    Test RMSE: ${metrics['RMSE']:,.2f} | Test MAE: ${metrics['MAE']:,.2f}")

    # XGBoost
    xgb_model, xgb_params, xgb_cv_rmse = tune_xgboost(X_train, y_train)
    if xgb_model is not None:
        save_tuned_model(xgb_model, xgb_params, xgb_cv_rmse, 'XGBoost', split_name, feature_cols)
        metrics, preds = evaluate_on_test(xgb_model, X_test, y_test, 'XGBoost_Tuned', split_name)
        tuned_results.append(metrics)
        tuned_models['XGBoost_Tuned'] = {'model': xgb_model, 'preds': preds, 'params': xgb_params}
        print(f"    Test RMSE: ${metrics['RMSE']:,.2f} | Test MAE: ${metrics['MAE']:,.2f}")

    # Save comparison table
    if previous_metrics:
        save_comparison(previous_metrics, tuned_results, split_name)

    return tuned_models, tuned_results


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------
def tuning_pipeline():
    """
    Run hyperparameter tuning on all three splits.
    Returns tuned model results.
    """
    print("=" * 60)
    print("STARTING HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"CV Strategy: TimeSeriesSplit({N_SPLITS_CV})")
    print(f"Search: RandomizedSearchCV ({N_ITER} iterations)")
    print("=" * 60)

    all_results = {}

    for split_name, (train_path, test_path) in SPLITS.items():
        models, metrics = tune_split(split_name, train_path, test_path)
        all_results[split_name] = {'models': models, 'metrics': metrics}

    # Save combined tuned metrics
    os.makedirs("outputs", exist_ok=True)
    all_metrics = []
    for split_name, data in all_results.items():
        all_metrics.extend(data['metrics'])

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv("outputs/metrics_tuned_all_splits.csv", index=False)
        print(f"\nCombined tuned metrics saved: outputs/metrics_tuned_all_splits.csv")

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)

    return all_results


# ---------------------------------------------------------
# RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    results = tuning_pipeline()
