# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 7: Model Evaluation & Diagnostics
# Loads predictions, generates evaluation reports + plots
# ---------------------------------------------------------

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SPLITS = ['70_30', '80_20', '90_10']
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------
# LOAD PREDICTIONS
# ---------------------------------------------------------
def load_predictions(split_name):
    """Load predictions CSV for a split."""
    path = f"outputs/predictions_{split_name}.csv"
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        return None
    return pd.read_csv(path)


def load_metrics(split_name):
    """Load metrics CSV for a split."""
    path = f"outputs/metrics_{split_name}.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------
# COMPUTE DETAILED METRICS
# ---------------------------------------------------------
def compute_detailed_metrics(df, model_col='Ensemble_Prediction'):
    """
    Compute detailed metrics for a single model's predictions.
    If the predictions file contains train + test rows, evaluate only test rows.
    """
    if 'Data_Split' in df.columns:
        test_df = df[df['Data_Split'] == 'Test'].copy()
        if not test_df.empty:
            df = test_df

    y_true = df['Actual']
    y_pred = df[model_col]

    residuals = y_true - y_pred
    pct_errors = (residuals / (y_true + 1e-8)) * 100

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs(pct_errors)),
        'Max_Error': np.max(np.abs(residuals)),
        'Median_Abs_Error': np.median(np.abs(residuals)),
        'Mean_Residual': np.mean(residuals),
        'Std_Residual': np.std(residuals),
        'Underprediction_%': np.mean(residuals > 0) * 100,
        'Overprediction_%': np.mean(residuals < 0) * 100,
    }


# ---------------------------------------------------------
# EVALUATE ALL MODELS ACROSS ALL SPLITS
# ---------------------------------------------------------
def evaluate_all_models():
    """
    Evaluate every model on every split and compile master table.
    """
    master_results = []

    for split in SPLITS:
        df = load_predictions(split)
        if df is None:
            continue

        pred_cols = [c for c in df.columns if '_Prediction' in c]

        for col in pred_cols:
            model_name = col.replace('_Prediction', '')
            metrics = compute_detailed_metrics(df, col)
            metrics['split'] = split
            metrics['model'] = model_name
            master_results.append(metrics)

    if not master_results:
        print("No prediction files found. Run train_models.py first.")
        return None

    master_df = pd.DataFrame(master_results)
    master_df = master_df.sort_values(['split', 'RMSE'])

    path = "outputs/evaluation_master.csv"
    master_df.to_csv(path, index=False)
    print(f"Master evaluation saved: {path}")
    return master_df


# ---------------------------------------------------------
# MODEL RANKING REPORT
# ---------------------------------------------------------
def generate_model_ranking(master_df):
    """
    Rank models by RMSE per split, identify best overall.
    """
    if master_df is None:
        return

    print("\n" + "=" * 60)
    print("MODEL RANKING BY SPLIT (RMSE)")
    print("=" * 60)

    for split in SPLITS:
        subset = master_df[master_df['split'] == split]
        if subset.empty:
            continue
        print(f"\nSplit: {split}")
        for _, row in subset.iterrows():
            print(f"  {row['model']:20s} | RMSE: ${row['RMSE']:>10,.2f} | MAE: ${row['MAE']:>10,.2f} | R²: {row['R2']:.4f}")

    print("\n" + "=" * 60)
    print("BEST MODEL PER SPLIT")
    print("=" * 60)
    best_per_split = master_df.loc[master_df.groupby('split')['RMSE'].idxmin()]
    for _, row in best_per_split.iterrows():
        print(f"  {row['split']:6s} → {row['model']:20s} (RMSE: ${row['RMSE']:,.2f})")

    return best_per_split


# ---------------------------------------------------------
# PLOT: PREDICTED VS ACTUAL
# ---------------------------------------------------------
def plot_predicted_vs_actual(df, split_name, model_col='Ensemble_Prediction', save=True):
    """
    Scatter plot: Predicted vs Actual with ideal line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    y_true = df['Actual']
    y_pred = df[model_col]
    model_name = model_col.replace('_Prediction', '')

    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', s=80)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Quarterly Claims ($)', fontsize=12)
    ax.set_ylabel('Predicted Quarterly Claims ($)', fontsize=12)
    ax.set_title(f'{model_name} — {split_name}: Predicted vs Actual', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        path = f"{PLOT_DIR}/predicted_vs_actual_{split_name}_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# PLOT: RESIDUALS DISTRIBUTION
# ---------------------------------------------------------
def plot_residuals(df, split_name, model_col='Ensemble_Prediction', save=True):
    """
    Histogram of residuals (Actual - Predicted).
    """
    residuals = df['Actual'] - df[model_col]
    model_name = model_col.replace('_Prediction', '')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Residual (Actual - Predicted) ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{model_name} — {split_name}: Residual Distribution', fontsize=14)
    ax.legend()

    if save:
        path = f"{PLOT_DIR}/residuals_{split_name}_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# PLOT: TIME-SERIES PREDICTED VS ACTUAL
# ---------------------------------------------------------
def plot_time_series(df, split_name, model_col='Ensemble_Prediction', save=True):
    """
    Line plot: Actual vs Predicted over quarters.
    """
    model_name = model_col.replace('_Prediction', '')

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    ax.plot(x, df['Actual'], 'o-', label='Actual', color='black', linewidth=2, markersize=8)
    ax.plot(x, df[model_col], 's--', label='Predicted', color='darkorange', linewidth=2, markersize=8)

    if 'Quarter_Label' in df.columns:
        ax.set_xticks(x)
        ax.set_xticklabels(df['Quarter_Label'], rotation=45, ha='right')

    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('Quarterly Claims ($)', fontsize=12)
    ax.set_title(f'{model_name} — {split_name}: Time-Series Forecast', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        path = f"{PLOT_DIR}/timeseries_{split_name}_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# PLOT: MODEL COMPARISON BAR CHART
# ---------------------------------------------------------
def plot_model_comparison_bar(master_df, metric='RMSE', save=True):
    """
    Grouped bar chart comparing all models across all splits.
    """
    if master_df is None or master_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    models = master_df['model'].unique()
    x = np.arange(len(models))
    width = 0.25

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, split in enumerate(SPLITS):
        subset = master_df[master_df['split'] == split].set_index('model').reindex(models)
        values = subset[metric].fillna(0).values
        ax.bar(x + i * width, values, width, label=split, color=colors[i], edgecolor='black')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison: {metric} Across Splits', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if save:
        path = f"{PLOT_DIR}/model_comparison_{metric}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# GENERATE ALL PLOTS FOR A SPLIT
# ---------------------------------------------------------
def generate_all_plots_for_split(split_name, model_cols=None):
    """
    Generate all evaluation plots for a single split.
    """
    df = load_predictions(split_name)
    if df is None:
        return

    if model_cols is None:
        model_cols = [c for c in df.columns if '_Prediction' in c]

    print(f"\n--- Generating plots for {split_name} ({len(model_cols)} models) ---")

    for col in model_cols:
        plot_predicted_vs_actual(df, split_name, col, save=True)
        plot_residuals(df, split_name, col, save=True)
        plot_time_series(df, split_name, col, save=True)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------
def extract_feature_importance(model_path, feature_cols_path):
    """
    Extract feature importance from trained tree-based models.
    """
    if not os.path.exists(model_path) or not os.path.exists(feature_cols_path):
        return None

    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    model = model_dict['model']

    if not hasattr(model, 'feature_importances_'):
        return None

    with open(feature_cols_path, 'r') as f:
        feature_cols = f.read().strip().split('\n')

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance


def plot_feature_importance(importance_df, model_name, split_name, top_n=15, save=True):
    """
    Horizontal bar chart of top N feature importances.
    """
    if importance_df is None or importance_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance_df.head(top_n).sort_values('importance', ascending=True)

    ax.barh(top['feature'], top['importance'], color='teal', edgecolor='black')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'{model_name} — {split_name}: Top {top_n} Features', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    if save:
        path = f"{PLOT_DIR}/feature_importance_{split_name}_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()
    else:
        plt.show()


def generate_all_feature_importance_plots(generate_plots=True):
    """
    Generate feature importance plots for all splits.
    """
    print("\n--- Generating Feature Importance Outputs ---")

    for split in SPLITS:
        for model_name in ['RandomForest', 'XGBoost']:
            model_path = f"models/{split}/{model_name}.pkl"
            feat_path = f"models/{split}/feature_columns.txt"

            importance = extract_feature_importance(model_path, feat_path)
            if importance is not None:
                if generate_plots:
                    plot_feature_importance(importance, model_name, split, save=True)

                csv_path = f"outputs/feature_importance_{split}_{model_name}.csv"
                importance.to_csv(csv_path, index=False)
                print(f"  Saved CSV: {csv_path}")


# ---------------------------------------------------------
# EVALUATION REPORT (Text Summary)
# ---------------------------------------------------------
def generate_evaluation_report(master_df):
    """
    Save a text summary report of all evaluations.
    """
    if master_df is None:
        return

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("INSURANCE CLAIM PREDICTION — MODEL EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")

    for split in SPLITS:
        subset = master_df[master_df['split'] == split]
        if subset.empty:
            continue

        report_lines.append(f"SPLIT: {split}")
        report_lines.append("-" * 40)

        for _, row in subset.iterrows():
            report_lines.append(
                f"  {row['model']:20s} | RMSE: ${row['RMSE']:>10,.2f} | "
                f"MAE: ${row['MAE']:>10,.2f} | R²: {row['R2']:.4f} | "
                f"MAPE: {row['MAPE']:.2f}%"
            )
        report_lines.append("")

    best = master_df.loc[master_df['RMSE'].idxmin()]
    report_lines.append("=" * 60)
    report_lines.append("BEST MODEL OVERALL")
    report_lines.append("=" * 60)
    report_lines.append(f"  Model: {best['model']}")
    report_lines.append(f"  Split: {best['split']}")
    report_lines.append(f"  RMSE:  ${best['RMSE']:,.2f}")
    report_lines.append(f"  MAE:   ${best['MAE']:,.2f}")
    report_lines.append(f"  R²:    {best['R2']:.4f}")
    report_lines.append("")

    report = "\n".join(report_lines)

    path = "outputs/evaluation_report.txt"
    with open(path, 'w') as f:
        f.write(report)
    print(f"Evaluation report saved: {path}")
    return report


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------
def evaluate_pipeline(generate_plots=True):
    """
    Full evaluation pipeline.
    Generates metrics, rankings, optional plots, and feature importance.
    """
    print("=" * 60)
    print("STARTING MODEL EVALUATION")
    print("=" * 60)

    master_df = evaluate_all_models()

    if master_df is None:
        print("No data to evaluate. Exiting.")
        return None

    print(f"\nEvaluated {len(master_df)} model-split combinations.")

    best_per_split = generate_model_ranking(master_df)

    if generate_plots:
        plot_model_comparison_bar(master_df, metric='RMSE', save=True)
        plot_model_comparison_bar(master_df, metric='MAE', save=True)
        plot_model_comparison_bar(master_df, metric='R2', save=True)

        for split in SPLITS:
            generate_all_plots_for_split(split)
    else:
        print("\nSkipping PNG plot generation. CSV/model outputs will still be generated.")

    generate_all_feature_importance_plots(generate_plots=generate_plots)

    report = generate_evaluation_report(master_df)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    if generate_plots:
        print(f"Plots saved to: {PLOT_DIR}/")
    else:
        print("PNG plots were not generated for this run.")
    print(f"Reports saved to: outputs/evaluation_report.txt")
    print(f"Master metrics: outputs/evaluation_master.csv")
    print("=" * 60)

    return master_df, report


# ---------------------------------------------------------
# RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    results = evaluate_pipeline()
