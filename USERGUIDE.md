## Purpose

Use this package to test the project from the beginning without relying on previously generated model files or output files.

The supervisor should be able to:

- Run the Streamlit website
- Upload a dataset
- Map dataset columns
- Validate uploaded data
- Run the full machine learning pipeline
- Generate new outputs and trained models
- Review all dashboard pages

## How To Run

1. Open this folder in a terminal.

```bash
cd "Insurance_Analytics_Start_From_Zero"
```

2. Install the required Python packages.

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib seaborn openpyxl
```

3. Run the website.

```bash
streamlit run web.py
```

4. In the website, open `Upload Data`.

5. Upload one of the sample datasets from the `test_mapping_files` folder or use:

```text
insurance_5years_full_formatted.xlsx
```

6. Confirm the detected file or sheet.

7. Review and confirm the column mapping.

8. Click `Run Full Pipeline`.

9. After the pipeline finishes, check:

- Quarterly Trends
- Customer Risk
- Incident Patterns
- Claim Breakdown
- Predictions & Risk
- Next Quarter Forecast
- Model Comparison
- What-If Simulator

## Recommended Test Order

### Test 1: Normal Full Dataset

Use:

```text
insurance_5years_full_formatted.xlsx
```

Expected result:

- Pipeline should run fully.
- Dashboard pages should be generated.
- Predictions and model comparison should be available.

### Test 2: Column Mapping Test

Use:

```text
test_mapping_files/01_standard_columns.csv
test_mapping_files/02_alias_columns.csv
test_mapping_files/03_required_plus_some_optional.csv
```

Expected result:

- System should detect required fields.
- User can confirm or adjust column mapping.
- Files with different column names should still be usable after mapping.

### Test 3: Missing Required Field Test

Use:

```text
test_mapping_files/04_missing_required_amount.csv
```

Expected result:

- System should warn that required data is missing.
- The pipeline should not continue until required fields are mapped or available.

### Test 4: Multi-Sheet Excel Test

Use:

```text
test_mapping_files/05_multi_sheet_mapping_test.xlsx
```

Expected result:

- System should detect available sheets.
- User should select only the valid data sheet or sheets.

### Test 5: Append Mode Test

Use:

```text
test_mapping_files/06_append_file1_12_standard_columns.csv
test_mapping_files/07_append_file2_alias_plus_extra_columns.csv
```

Expected result:

- System should combine similar claim records from multiple files.
- Column names may be different, but mapping should standardize them.

### Test 6: Join Mode Test

Use:

```text
test_mapping_files/10_join_customer_policy_shared_key.csv
test_mapping_files/11_join_incident_claim_shared_key.csv
```

Expected result:

- User should choose a shared join key.
- System should merge related tables using the selected key.

## Files Included

- `web.py`
- `preprocessing.py`
- `feature_engineering.py`
- `aggregation.py`
- `split_data.py`
- `train_models.py`
- `tuning.py`
- `evaluate_models.py`
- `predict.py`
- `README.md`
- `icons/`
- `test_mapping_files/`
- `insurance_5years_full_formatted.xlsx`

## Files Not Included On Purpose

The following generated folders are not required for start-from-zero testing:

- `outputs/`
- `models/`
- `__pycache__/`

These files should be generated again after running the full pipeline.

