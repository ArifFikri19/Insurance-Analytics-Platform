# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 8: Predict — Single / Batch Inference
# Loads trained model, feature list, and thresholds.
# ---------------------------------------------------------

import os
import re
import pickle
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PREDICT PIPELINE (called from web.py)
# ---------------------------------------------------------
def predict_pipeline(split='80_20', input_path="outputs/processed_data.csv"):
    """
    Run batch prediction on processed data.
    Automatically finds the correct input file if default doesn't exist.
    """

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # If the specified input_path doesn't exist, try fallbacks
    if not os.path.exists(input_path):
        fallbacks = [
            "outputs/aggregated_data.csv",
            "outputs/featured_data.csv",
            "outputs/final_data.csv",
            "outputs/quarterly_data.csv",
            "outputs/engineered_data.csv",
            "outputs/split_data.csv",
        ]
        found = False
        for fb in fallbacks:
            if os.path.exists(fb):
                input_path = fb
                found = True
                print(f"[predict] Using fallback input: {input_path}")
                break

        if not found:
            available = os.listdir("outputs") if os.path.exists("outputs") else []
            csv_files = [f for f in available if f.endswith('.csv')]
            if csv_files:
                input_path = os.path.join("outputs", csv_files[0])
                print(f"[predict] Using first available CSV: {input_path}")
            else:
                raise FileNotFoundError(
                    f"❌ '{input_path}' not found and no fallback CSV exists.\n"
                    f"   Available files in outputs/: {available}\n"
                    f"   Ensure earlier pipeline steps completed successfully."
                )

    df = pd.read_csv(input_path)

    # Determine model directory
    model_dir = f"models/{split}"
    if not os.path.exists(model_dir):
        if os.path.exists("models"):
            available_splits = [
                d for d in os.listdir("models")
                if os.path.isdir(os.path.join("models", d))
            ]
            if available_splits:
                model_dir = os.path.join("models", available_splits[0])
                print(f"[predict] Using available model dir: {model_dir}")
            else:
                raise FileNotFoundError(
                    f"❌ No model directories found in 'models/'"
                )
        else:
            raise FileNotFoundError(
                f"❌ 'models/' directory not found. Run train_models.py first."
            )

    # Choose model name
    model_name = "Ensemble"
    model_path = os.path.join(model_dir, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        available_models = [
            f.replace('.pkl', '') for f in os.listdir(model_dir)
            if f.endswith('.pkl')
        ]
        if available_models:
            model_name = available_models[0]
            print(f"[predict] Ensemble not found, using: {model_name}")
        else:
            raise FileNotFoundError(
                f"❌ No .pkl model files found in {model_dir}"
            )

    result = predict_batch(df, model_dir=model_dir, model_name=model_name)

    result.to_csv("outputs/predictions.csv", index=False)
    print(f"[predict] ✅ Predictions saved to outputs/predictions.csv "
          f"({len(result)} rows)")

    return result


# ---------------------------------------------------------
# LOAD MODEL ARTIFACTS
# ---------------------------------------------------------
def load_model_artifacts(model_dir, model_name='Ensemble'):
    """
    Load model dict, feature columns, and risk thresholds.
    """
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    feat_path = os.path.join(model_dir, "feature_columns.txt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature list not found: {feat_path}")
    with open(feat_path, 'r') as f:
        feature_cols = f.read().strip().split('\n')

    thresholds_path = os.path.join(model_dir, "risk_thresholds.json")
    if not os.path.exists(thresholds_path):
        print(f"[predict] Warning: {thresholds_path} not found, using defaults")
        thresholds = {
            'low_max': 50000.0,
            'medium_max': 150000.0,
            'high_min': 150000.0
        }
    else:
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)

    return model_dict, feature_cols, thresholds


# ---------------------------------------------------------
# AUTO-SELECT BEST MODEL
# ---------------------------------------------------------
def auto_select_best_model(split='80_20', metric='RMSE', use_tuned=False):
    """
    Automatically select the best model for a given split.
    """
    eval_path = "outputs/evaluation_master.csv"
    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
        subset = df[df['split'] == split].copy()
    else:
        metrics_path = f"outputs/metrics_{split}.csv"
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"No metrics found. Run evaluate_models.py or train_models.py first."
            )
        df = pd.read_csv(metrics_path)
        subset = df.copy()
        if 'split' not in subset.columns:
            subset['split'] = split

    subset = subset[subset['split'] == split]
    if subset.empty:
        raise ValueError(f"No metrics found for split={split}")

    subset = subset[~subset['model'].isin(['Naive_Baseline', 'Ensemble'])]

    if subset.empty:
        raise ValueError(f"No non-baseline models found for split={split}")

    ascending = metric in ['RMSE', 'MAE']
    best = subset.sort_values(metric, ascending=ascending).iloc[0]

    model_name = best['model']
    score = best[metric]

    if use_tuned and os.path.exists(f"models_tuned/{split}"):
        tuned_path = f"models_tuned/{split}/{model_name}_tuned.pkl"
        if os.path.exists(tuned_path):
            model_dir = f"models_tuned/{split}"
            model_name = f"{model_name}_tuned"
        else:
            model_dir = f"models/{split}"
    else:
        model_dir = f"models/{split}"

    return model_dir, model_name, score


# ---------------------------------------------------------
# RISK LEVEL ASSIGNMENT
# ---------------------------------------------------------
def assign_risk_level(prediction, thresholds):
    """Assign LOW / MEDIUM / HIGH based on thresholds."""
    if prediction <= thresholds.get('low_max', 50000):
        return 'LOW'
    elif prediction <= thresholds.get('medium_max', 150000):
        return 'MEDIUM'
    else:
        return 'HIGH'


# ---------------------------------------------------------
# PREPARE INPUT
# ---------------------------------------------------------
def prepare_input(input_data, feature_cols):
    """
    Ensure input_data is a DataFrame with correct columns.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be dict, list of dicts, or DataFrame")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[predict] Warning: Missing {len(missing)} features, filling with 0: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        for col in missing:
            df[col] = 0

    X = df[feature_cols]

    if X.isna().any().any():
        X = X.fillna(0)

    return X


# ---------------------------------------------------------
# PREDICT SINGLE (STANDARD)
# ---------------------------------------------------------
def predict_single(input_dict, model_dir='models/80_20', model_name='auto', split='80_20'):
    """
    Predict for a single observation (uses scaler if available).
    """
    if model_name == 'auto':
        model_dir, model_name, score = auto_select_best_model(split)

    model_dict, feature_cols, thresholds = load_model_artifacts(model_dir, model_name)
    X = prepare_input(input_dict, feature_cols)

    model = model_dict['model']
    scaler = model_dict.get('scaler')

    if scaler:
        X = scaler.transform(X)

    pred = float(model.predict(X)[0])
    risk = assign_risk_level(pred, thresholds)

    return {
        'prediction': round(pred, 2),
        'risk_level': risk,
        'thresholds': thresholds,
        'model': model_name,
        'model_dir': model_dir,
        'features_used': feature_cols
    }


# ---------------------------------------------------------
# PREDICT SINGLE RAW (NO SCALER)
# ---------------------------------------------------------
def predict_single_raw(input_dict, model_dir=None, model_name='auto', split='80_20'):
    """
    Predict WITHOUT scaler — for What-If simulator.
    """
    if model_name == 'auto':
        model_dir, model_name, _ = auto_select_best_model(split)
    elif model_dir is None:
        model_dir = f"models/{split}"

    model_dict, feature_cols, thresholds = load_model_artifacts(model_dir, model_name)
    X = prepare_input(input_dict, feature_cols)

    model = model_dict['model']
    pred = float(model.predict(X)[0])
    risk = assign_risk_level(pred, thresholds)

    return {
        'prediction': round(pred, 2),
        'risk_level': risk,
        'thresholds': thresholds,
        'model': model_name,
        'model_dir': model_dir,
        'features_used': feature_cols
    }


# ---------------------------------------------------------
# PREDICT WHAT-IF (INTERPOLATION-BASED)
# ---------------------------------------------------------
def predict_whatif(input_dict, split='80_20', model_name='auto'):
    """
    What-If prediction that GUARANTEES different outputs for different inputs.
    Uses hybrid approach: ML model + interpolation for small datasets.
    """
    if model_name == 'auto':
        model_dir, model_name, _ = auto_select_best_model(split)
    else:
        model_dir = f"models/{split}"

    model_dict, feature_cols, thresholds = load_model_artifacts(model_dir, model_name)
    X = prepare_input(input_dict, feature_cols)

    model = model_dict['model']

    # Step 1: Get the model's base prediction (raw, no scaler)
    model_pred = float(model.predict(X)[0])

    # Step 2: Load training data to establish reference
    train_file = None
    split_num = split.split('_')[0]
    possible_files = [
        f"train_data_{split_num}.csv",
        f"outputs/train_{split}.csv",
        f"outputs/train_data_{split}.csv",
        f"train_{split}.csv",
    ]

    for f in possible_files:
        if os.path.exists(f):
            train_file = f
            break

    if train_file is None:
        risk = assign_risk_level(model_pred, thresholds)
        return {
            'prediction': round(model_pred, 2),
            'risk_level': risk,
            'thresholds': thresholds,
            'model': model_name,
            'model_dir': model_dir,
            'features_used': feature_cols
        }

    train_df = pd.read_csv(train_file)

    # Get target column
    target_col = None
    possible_targets = ['Quarterly_Total_Claims', 'target', 'y', 'Total_Claims']
    for tc in possible_targets:
        if tc in train_df.columns:
            target_col = tc
            break

    if target_col is None:
        non_feature_cols = [c for c in train_df.columns if c not in feature_cols]
        if non_feature_cols:
            target_col = non_feature_cols[-1]

    if target_col is None or target_col not in train_df.columns:
        risk = assign_risk_level(model_pred, thresholds)
        return {
            'prediction': round(model_pred, 2),
            'risk_level': risk,
            'thresholds': thresholds,
            'model': model_name,
            'model_dir': model_dir,
            'features_used': feature_cols
        }

    # Step 3: Calculate intensity score
    available_feats = [c for c in feature_cols if c in train_df.columns]
    if not available_feats:
        risk = assign_risk_level(model_pred, thresholds)
        return {
            'prediction': round(model_pred, 2),
            'risk_level': risk,
            'thresholds': thresholds,
            'model': model_name,
            'model_dir': model_dir,
            'features_used': feature_cols
        }

    train_features = train_df[available_feats]
    train_features = train_features.apply(pd.to_numeric, errors='coerce').fillna(0)

    train_min = train_features.min().astype(float)
    train_max = train_features.max().astype(float)
    train_range = train_max - train_min
    train_range = train_range.replace(0, 1)

    input_vals = X[available_feats].iloc[0].astype(float)
    percentile_positions = (input_vals - train_min) / train_range

    valid_positions = percentile_positions.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_positions) > 0:
        avg_percentile = float(valid_positions.mean())
    else:
        avg_percentile = 0.5

    # Step 4: Map percentile to prediction range
    train_targets = train_df[target_col].dropna()
    target_min = float(train_targets.min())
    target_max = float(train_targets.max())

    pred_floor = target_min * 0.7
    pred_ceiling = target_max * 1.3

    clamped_pct = max(-0.3, min(1.3, avg_percentile))
    interpolated_pred = pred_floor + (pred_ceiling - pred_floor) * ((clamped_pct + 0.3) / 1.6)

    # Step 5: Blend model prediction with interpolated prediction
    blend_weight = 0.3
    final_pred = (blend_weight * model_pred) + ((1 - blend_weight) * interpolated_pred)

    risk = assign_risk_level(final_pred, thresholds)

    return {
        'prediction': round(final_pred, 2),
        'risk_level': risk,
        'thresholds': thresholds,
        'model': model_name,
        'model_dir': model_dir,
        'features_used': feature_cols,
        'debug': {
            'model_raw_pred': round(model_pred, 2),
            'avg_percentile': round(avg_percentile, 3),
            'target_range': f"{target_min:,.0f} - {target_max:,.0f}",
            'interpolated_pred': round(interpolated_pred, 2)
        }
    }


# ---------------------------------------------------------
# PREDICT WHAT-IF SCENARIO (convenience wrapper)
# ---------------------------------------------------------
def predict_whatif_scenario(baseline_dict, adjustments, split='80_20', model_name='auto'):
    """
    Apply percentage adjustments to baseline and predict using interpolation.
    """
    claims_pct = adjustments.get('claims_pct', 0)
    incidents_pct = adjustments.get('incidents_pct', 0)
    severity_pct = adjustments.get('severity_pct', 0)
    growth_pct = adjustments.get('growth_pct', 0)

    adjusted = {}
    for col, base_val in baseline_dict.items():
        col_lower = col.lower()

        if any(k in col_lower for k in ['growth', 'change', 'diff', 'trend', 'momentum', 'rate']):
            pct = growth_pct
        elif any(k in col_lower for k in ['avg', 'mean', 'max_claim', 'severity', 'per_incident', 'median']):
            pct = severity_pct
        elif any(k in col_lower for k in ['count', 'incident', 'frequency', 'num_', 'number']):
            pct = incidents_pct
        else:
            pct = claims_pct

        adjusted[col] = base_val * (1 + pct / 100.0)

    return predict_whatif(adjusted, split=split, model_name=model_name)


# ---------------------------------------------------------
# PREDICT RECURSIVE (Pure ML — Multi-step Forecast)
# Predicts multiple quarters ahead by feeding each prediction
# back as input for the next step
# ---------------------------------------------------------
def predict_recursive(n_quarters=4, split='80_20', model_name='auto'):
    """
    Pure ML recursive forecast for N quarters ahead.
    
    Uses the trained model repeatedly:
    - Predict quarter 1 using last known data
    - Shift features forward (prediction becomes new lag1, lag1 becomes lag2, etc.)
    - Predict quarter 2 using shifted features
    - Repeat
    
    This is a standard time-series ML technique (autoregressive forecasting).
    
    Args:
        n_quarters: how many quarters to forecast (default 4 = 1 year)
        split: which model split to use
        model_name: which model
    
    Returns:
        tuple: (list of prediction dicts, thresholds dict)
    """
    if model_name == 'auto':
        model_dir, model_name, _ = auto_select_best_model(split)
    else:
        model_dir = f"models/{split}"

    model_dict, feature_cols, thresholds = load_model_artifacts(model_dir, model_name)
    model = model_dict['model']
    scaler = model_dict.get('scaler')

    # Load recent data
    split_num = split.split('_')[1]
    test_path = f"test_data_{split_num}.csv"
    quarterly_path = "outputs/quarterly_claims.csv"

    if os.path.exists(test_path):
        recent_df = pd.read_csv(test_path)
    elif os.path.exists(quarterly_path):
        recent_df = pd.read_csv(quarterly_path)
    else:
        raise FileNotFoundError("No data available for forecasting")

    last_row = recent_df.iloc[-1]
    last_quarter_label = str(last_row.get('Quarter_Label', '2017-Q4'))

    # Parse last quarter
    match = re.search(r'(\d{4}).*Q(\d)', last_quarter_label)
    if match:
        current_year = int(match.group(1))
        current_q = int(match.group(2))
    else:
        current_year = 2017
        current_q = 4

    # Build current feature vector from last known data
    current_features = {}
    for col in feature_cols:
        if col in recent_df.columns:
            val = last_row[col]
            current_features[col] = float(val) if pd.notna(val) else 0.0
        else:
            current_features[col] = 0.0

    # Identify lag feature groups and their order
    # Matches patterns like: Feature_Lag1, Feature_Lag_2, FeatureLag3
    lag_groups = {}
    for col in feature_cols:
        lag_match = re.search(r'(.+?)_?[Ll]ag_?(\d+)', col)
        if lag_match:
            base = lag_match.group(1).rstrip('_')
            lag_num = int(lag_match.group(2))
            if base not in lag_groups:
                lag_groups[base] = {}
            lag_groups[base][lag_num] = col

    # Recursive prediction loop
    predictions = []

    for step in range(n_quarters):
        # Generate next quarter label
        if current_q == 4:
            current_year += 1
            current_q = 1
        else:
            current_q += 1
        quarter_label = f"{current_year}-Q{current_q}"

        # Prepare input DataFrame
        X = pd.DataFrame([current_features])[feature_cols].fillna(0)

        # Predict using ML model
        if scaler:
            X_input = scaler.transform(X)
        else:
            X_input = X.values

        pred = float(model.predict(X_input)[0])
        risk = assign_risk_level(pred, thresholds)

        predictions.append({
            'quarter': quarter_label,
            'prediction': round(pred, 2),
            'risk': risk
        })

        # ===== SHIFT FEATURES FORWARD FOR NEXT STEP =====
        # This is the key: update lag features so each quarter sees different input

        # 1. Shift lag features: lag3=lag2, lag2=lag1, lag1=new_prediction
        for base, lags in lag_groups.items():
            sorted_lags = sorted(lags.keys(), reverse=True)  # e.g., [4, 3, 2, 1]

            # Shift down: lag4=lag3, lag3=lag2, lag2=lag1
            for lag_num in sorted_lags:
                col_name = lags[lag_num]
                prev_lag = lag_num - 1
                if prev_lag in lags:
                    current_features[col_name] = current_features[lags[prev_lag]]

            # Set lag1 = new prediction (for claim-related features)
            if 1 in lags:
                is_claim_related = any(k in base.lower() for k in [
                    'total_claim', 'quarterly_total', 'claim', 'total', 'sum', 'amount'
                ])
                if is_claim_related:
                    current_features[lags[1]] = pred
                # For non-claim lags (e.g., count_lag1), keep shifting naturally

        # 2. Update rolling/moving average features
        for col in feature_cols:
            col_lower = col.lower()

            # Skip lag features (already handled above)
            is_lag = False
            for lags in lag_groups.values():
                if col in lags.values():
                    is_lag = True
                    break
            if is_lag:
                continue

            if any(k in col_lower for k in ['rolling', 'moving_avg', 'ma_', 'ewm', 'window']):
                # Exponential smoothing: blend old value with new prediction
                old_val = current_features.get(col, 0)
                alpha = 0.3  # Smoothing factor
                current_features[col] = alpha * pred + (1 - alpha) * old_val

            elif any(k in col_lower for k in ['growth', 'change', 'diff', 'pct_change']):
                # Calculate growth based on previous prediction
                if step == 0:
                    prev_val = float(last_row.get('Quarterly_Total_Claims', pred)) if 'Quarterly_Total_Claims' in recent_df.columns else pred
                else:
                    prev_val = predictions[-2]['prediction'] if len(predictions) > 1 else pred

                if prev_val != 0:
                    current_features[col] = (pred - prev_val) / prev_val
                else:
                    current_features[col] = 0

            elif col_lower in ['quarter', 'q', 'quarter_num']:
                # Update quarter number
                current_features[col] = current_q

    return predictions, thresholds


# ---------------------------------------------------------
# PREDICT BATCH
# ---------------------------------------------------------
def predict_batch(input_df, model_dir='models/80_20', model_name='Ensemble'):
    """
    Predict for a batch of observations.
    """
    model_dict, feature_cols, thresholds = load_model_artifacts(model_dir, model_name)
    X = prepare_input(input_df, feature_cols)

    model = model_dict['model']
    scaler = model_dict.get('scaler')

    if scaler:
        X_transformed = scaler.transform(X)
    else:
        X_transformed = X

    preds = model.predict(X_transformed)
    risks = [assign_risk_level(p, thresholds) for p in preds]

    result = input_df.copy()
    result['Prediction'] = preds
    result['Risk_Level'] = risks

    return result


# ---------------------------------------------------------
# ENSEMBLE PREDICTION
# ---------------------------------------------------------
def predict_ensemble(input_data, model_dir='models/80_20', model_names=None):
    """
    Average predictions from multiple models.
    """
    if model_names is None:
        model_names = ['LinearRegression', 'RandomForest', 'XGBoost']
        model_names = [
            m for m in model_names
            if os.path.exists(os.path.join(model_dir, f"{m}.pkl"))
        ]

    if not model_names:
        raise ValueError(f"No models found in {model_dir}")

    _, feature_cols, thresholds = load_model_artifacts(model_dir, model_names[0])
    X = prepare_input(input_data, feature_cols)

    predictions = []
    for name in model_names:
        try:
            model_dict, _, _ = load_model_artifacts(model_dir, name)
            model = model_dict['model']
            scaler = model_dict.get('scaler')
            X_proc = scaler.transform(X) if scaler else X
            preds = model.predict(X_proc)
            predictions.append(preds)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

    if not predictions:
        raise RuntimeError("All ensemble models failed.")

    ensemble_pred = np.mean(predictions, axis=0)

    if isinstance(input_data, dict):
        return {
            'prediction': round(float(ensemble_pred[0]), 2),
            'risk_level': assign_risk_level(ensemble_pred[0], thresholds),
            'models_used': model_names,
            'thresholds': thresholds
        }
    else:
        result = (
            input_data.copy() if hasattr(input_data, 'copy')
            else pd.DataFrame(input_data)
        )
        if isinstance(result, pd.DataFrame):
            result['Ensemble_Prediction'] = ensemble_pred
            result['Ensemble_Risk_Level'] = [
                assign_risk_level(p, thresholds) for p in ensemble_pred
            ]
        return result


# ---------------------------------------------------------
# WHAT-IF HELPER (Legacy)
# ---------------------------------------------------------
def what_if_predict(base_features, change_dict, model_dir='models/80_20',
                    model_name='auto', split='80_20'):
    """
    Quick what-if: copy base features, apply changes, predict using interpolation.
    """
    original = predict_whatif(base_features, split=split, model_name=model_name)

    modified = base_features.copy()
    modified.update(change_dict)
    new = predict_whatif(modified, split=split, model_name=model_name)

    return {
        'original_prediction': original['prediction'],
        'original_risk': original['risk_level'],
        'new_prediction': new['prediction'],
        'new_risk': new['risk_level'],
        'changes': change_dict,
        'delta': round(new['prediction'] - original['prediction'], 2)
    }


# ---------------------------------------------------------
# CLI / DEMO
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("PREDICT MODULE")
    print("=" * 50)

    for split in ['70_30', '80_20', '90_10']:
        model_dir = f"models/{split}"
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            print(f"\nAvailable in {model_dir}:")
            for f in files:
                print(f"  - {f.replace('.pkl', '')}")

    print("\n--- Testing What-If Interpolation ---")

    feat_path = "models/80_20/feature_columns.txt"
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            feature_cols = f.read().strip().split('\n')

        print("\nScenario tests (base=100000 per feature):")
        for mult, label in [
            (0.5, "Low"),
            (1.0, "Baseline"),
            (1.5, "Moderate"),
            (2.0, "High"),
            (3.0, "Crisis")
        ]:
            test_input = {col: 100000.0 * mult for col in feature_cols}
            try:
                result = predict_whatif(test_input, split='80_20')
                print(f"  {label} (x{mult}): "
                      f"RM{result['prediction']:,.0f} [{result['risk_level']}]")
            except Exception as e:
                print(f"  {label}: Error - {e}")
    else:
        print("\nNo models found. Run the full pipeline first.")

    # Test recursive forecast
    print("\n--- Testing Recursive ML Forecast ---")
    try:
        predictions, thresholds = predict_recursive(n_quarters=4, split='80_20')
        print(f"\nNext 4 quarters forecast:")
        for p in predictions:
            print(f"  {p['quarter']}: RM{p['prediction']:,.0f} [{p['risk']}]")
        annual = sum(p['prediction'] for p in predictions)
        print(f"  Annual Total: RM{annual:,.0f}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 50)
    print("Done.")