# Insurance Analytics Platform

Insurance Analytics Platform is a Streamlit-based web application for insurance claim analytics, quarterly claim forecasting, model comparison, risk classification, and what-if simulation.

The system allows users to upload insurance datasets, map different column names, validate required fields, run a complete machine learning pipeline, and view the generated dashboard insights.

## Main Features

- Upload CSV or Excel insurance datasets
- Support multiple files and multiple Excel sheets
- Column mapping and source-by-source validation
- Data preprocessing and feature engineering
- Quarterly claim aggregation
- Train/test split by chronological order
- Machine learning model training and tuning
- Prediction and risk-level generation
- Quarterly trend analysis
- Customer risk analysis
- Incident pattern analysis
- Claim amount breakdown
- Model performance comparison
- Next-quarter and next-year forecasting
- What-if prediction simulator

## Machine Learning Models

The project uses regression models because the main target is numerical claim amount prediction.

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Ensemble Regressor
- Naive Baseline for benchmark comparison only

Risk levels such as LOW, MEDIUM, and HIGH are generated after prediction using rule-based thresholds.

## Project Structure

```text
Insurance Project/
├── web.py                     # Main Streamlit dashboard
├── preprocessing.py           # Data cleaning, column mapping, validation
├── feature_engineering.py     # Feature creation
├── aggregation.py             # Quarterly aggregation
├── split_data.py              # Chronological train/test split
├── train_models.py            # Model training
├── tuning.py                  # Hyperparameter tuning
├── evaluate_models.py         # Model evaluation
├── predict.py                 # Prediction generation
├── icons/                     # Logo and icon assets
├── outputs/                   # Generated output files
├── models/                    # Trained model files
├── test_mapping_files/        # Sample files for testing column mapping
└── requirements.txt           # Python dependencies
```

## Required Dataset Fields

The minimum fields needed to run the full machine learning pipeline are:

- Incident Date
- Total Claim Amount

Recommended fields for full dashboard functionality include:

- Customer ID
- Age
- Months As Customer
- Policy Annual Premium
- Policy Deductable
- Incident Type
- Incident Severity
- Injury Claim
- Property Claim
- Vehicle Claim

The uploaded dataset does not need to use the exact same column names. Users can map different source column names to the standard fields during upload.

Example:

```python
{
    "Incident Date": "Accident Date",
    "Total Claim Amount": "Claim Paid",
    "Age": "CustAge"
}
```

## Setup Instructions

1. Open a terminal in the project folder.

```bash
cd "D:\Insurance Project"
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the required packages.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is empty or incomplete, install the common required packages:

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib seaborn openpyxl
```

## How To Run The Website

Run the Streamlit application:

```bash
streamlit run web.py
```

The application will open in the browser. If it does not open automatically, copy the local URL shown in the terminal.

## How To Use The System

1. Open the website using `streamlit run web.py`.
2. Go to `Upload Data`.
3. Upload one or more CSV or Excel files.
4. Select the sheets or sources that contain insurance claim data.
5. Confirm column mapping for each selected file or sheet.
6. Run the full pipeline.
7. After the pipeline finishes, review the dashboard pages:
   - Quarterly Trends
   - Customer Risk
   - Incident Patterns
   - Claim Breakdown
   - Predictions & Risk
   - Next Quarter Forecast
   - Model Comparison
   - What-If Simulator

## Pipeline Order

The complete pipeline runs in this order:

1. Preprocess data
2. Engineer features
3. Aggregate quarters
4. Split train/test data
5. Train models
6. Tune models
7. Evaluate models
8. Generate predictions

## Important Output Files

The dashboard pages depend on generated output files.

```text
outputs/quarterly_claims.csv
outputs/customer_risk_scores.csv
outputs/quarterly_incident_patterns.csv
outputs/quarterly_claim_breakdown.csv
outputs/evaluation_master.csv
outputs/predictions_80_20.csv
```

If a page shows `Data Not Available`, run the complete pipeline from the Upload Data page.

## Current Limitations

- Prediction quality depends heavily on uploaded data quality.
- Limited historical data can reduce model reliability.
- Column mapping must be checked carefully before running the pipeline.
- The system does not currently include user login, audit logs, deployment setup, or backup/security features.
- Deep learning models such as LSTM are not used because the project is mainly based on structured tabular insurance data and limited quarterly history.

## Recommended Future Improvements

- Add a data quality report
- Add model performance report export
- Add SHAP or feature importance explanations
- Add user login and access control
- Add audit logging
- Add deployment configuration
- Add backup and security handling
- Add client-specific customization
- Add more historical data for stronger forecasting
