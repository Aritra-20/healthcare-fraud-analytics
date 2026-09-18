# Healthcare Provider Fraud Detection & Analytics

> **Live interactive dashboard:** https://public.tableau.com/views/Medicare_Fraud_Analytics/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

## Summary

Medicare fraud costs the system billions every year, but the raw data doesn't make it easy to catch. Claims sit at the *individual claim* level, while investigations happen at the *provider* level, and fraud tends to show up as behavior over time rather than a single odd claim.

This project rolls more than 500,000 Medicare inpatient and outpatient claims up into a dataset of 5,410 providers. For each one I built financial, behavioral and billing-trend features, then trained a Decision Tree and a Random Forest to flag providers likely to be committing fraud. Only about 9.4% of providers are labeled fraudulent, so most of the modeling decisions were about not letting that imbalance fool the results.

**Headline result:** the Random Forest reached a PR-AUC of **0.704** against **0.464** for the Decision Tree. At the default threshold it flagged providers with 75% precision and 50% recall. The details, including where the model is weak, are in [Results](#results).

---

## What the model found

Random Forest feature importances (9 features, 100 trees):

| Rank | Feature | Importance |
|---|---|---|
| 1 | Total_Reimbursed | 0.379 |
| 2 | Total_Deductible | 0.197 |
| 3 | Unique_Beneficiaries | 0.075 |
| 4 | Inpatient_Ratio | 0.074 |
| 5 | Avg_Claim_Duration | 0.064 |
| 6 | Total_Claims | 0.061 |
| 7 | Claims_Per_Bene | 0.054 |
| 8 | Billing_Trend_Deviation | 0.049 |
| 9 | Avg_Patient_Risk | 0.049 |

A few things stand out:

1. **Scale does most of the work.** Total reimbursement and total deductible together account for about 58% of the model's importance. Fraudulent providers, as labeled in this dataset, simply move a lot more money than benign ones. Total deductible is probably acting as a second measure of volume rather than telling us something separate.
2. **Behavioral ratios matter, but less.** Claims per beneficiary, average claim duration and the inpatient share each contribute roughly 5 to 7%. The idea behind Claims_Per_Bene is that repeatedly billing the same patients is a sign of overtreatment. In the Tableau dashboard the difference shows up visually, but the model leans on it far less than on raw dollar volume.
3. **The trend feature added little.** I expected Billing_Trend_Deviation to be a leading indicator, catching providers whose billing is accelerating away from their own history. In practice it ranks 8th of 9. The reasons are probably in how it's built (see [Limitations](#limitations)), and improving it is the most obvious next step.
4. **Patient chronic-condition burden ranked last.** I initially hypothesized that fraudulent providers target high-morbidity patients. This feature, on average per provider, didn't separate the classes much in the Random Forest.

---

## Data and methodology

### Data source

De-identified Medicare data from Kaggle, split across four relational tables: inpatient claims, outpatient claims, beneficiary details and provider fraud labels.

- **Source:** [Kaggle: Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
- Only the four `Train` files are used. The `Test` files have no fraud labels, so I hold out 20% of the labeled providers for validation instead.

### Feature engineering (claims to providers)

Inpatient and outpatient claims are stacked into one table, joined to beneficiary data, then grouped by provider. Missing reimbursement and deductible values are treated as $0.

| Feature | Description | Why it might matter |
|---|---|---|
| Total_Claims, Unique_Beneficiaries | Claim count and distinct patients per provider | Basic volume measures |
| Total_Reimbursed, Total_Deductible | Summed dollars per provider | Financial scale |
| Claims_Per_Bene | Total claims divided by unique beneficiaries | Flags repeat billing on the same patients |
| Avg_Patient_Risk | Number of chronic conditions per patient, averaged per provider | Flags targeting of high-morbidity patients |
| Avg_Claim_Duration | Mean claim length in days (same-day claims count as 1) | Flags unusually long stays |
| Inpatient_Ratio | Share of a provider's claims that are inpatient | Captures the provider's care mix |
| Billing_Trend_Deviation | Actual latest-month reimbursement vs. an exponential smoothing forecast, normalized by the forecast | Meant to flag billing that departs from the provider's own trend |

### Time-series forecasting

Reimbursements are summed into a monthly series for each provider. For providers with at least four months of history, I fit Simple Exponential Smoothing on every month except the last, forecast that last month, and compute `(actual - forecast) / forecast`. A positive score means the provider billed more than its own trend predicted. Providers with too little history get the median score instead of being dropped.

### Modeling

- **Models:** a Decision Tree (`max_depth=5`) as an interpretable baseline, and a Random Forest (100 trees).
- **Class imbalance:** roughly 90.6% benign to 9.4% fraud. Rather than oversample with SMOTE, which adds synthetic records, I used `class_weight='balanced'` on both models so that missing a fraudulent provider costs more.
- **Split:** one stratified 80/20 train/validation split (`random_state=42`), shared by both models. Validation has 1,082 providers, 101 of them fraudulent.
- **Metric:** accuracy is close to useless here, since predicting "benign" every time scores about 91%. I compare models on **precision-recall AUC**, which reflects the trade-off between catching fraud and sending auditors after innocent providers.

---

## Results

| Model | PR-AUC | Fraud precision | Fraud recall | Fraud F1 |
|---|---|---|---|---|
| Decision Tree | 0.464 | 0.37 | 0.88 | 0.52 |
| Random Forest | **0.704** | **0.75** | 0.50 | **0.60** |

Confusion matrices on the validation set (rows are actual, columns are predicted; benign first):

| | Random Forest | Decision Tree |
|---|---|---|
| Actual benign | 964 correct, 17 false alarms | 827 correct, 154 false alarms |
| Actual fraud | 50 missed, 51 caught | 12 missed, 89 caught |

The Random Forest wins on PR-AUC by 0.24, and it's the model I would use. But the table hides a real trade-off, and it is worth being clear about:

- The **Random Forest is conservative.** It flags 68 providers and 51 of them are actually fraudulent, which keeps wasted audits low, but it misses half of the fraud (50 of 101).
- The **Decision Tree is aggressive.** It catches 89 of 101 fraudulent providers, but it flags 243 to do it, and 154 of those are false alarms.

Which one is "better" depends on how much audit capacity is available. A team that can review many providers might prefer the Decision Tree's recall. Lowering the Random Forest's decision threshold would probably land somewhere useful in between, and that is the next thing I'd test.

---

## Limitations

- **One validation split.** All numbers come from a single 80/20 split with 101 fraud cases. Cross-validation would say how stable the 0.704 really is.
- **No tuning.** Both models use mostly default settings, and the decision threshold is the default 0.5.
- **The trend feature is thin.** It compares only the final month with a forecast, so one unusual month drives the whole score. Providers with short histories all receive the same median value. Rolling or windowed deviations, or the volatility of the series, would likely carry more signal.
- **Importances are impurity-based.** They tend to favor continuous, high-variance features, and correlated volume features (reimbursed, deductible, claims) split credit among themselves. Permutation importance or SHAP would be a more trustworthy read.
- **Labels are potential fraud flags,** not confirmed convictions, so the model learns whatever process produced those labels.

---

## Tech stack

- **Data:** Python, Pandas, NumPy
- **Forecasting:** Statsmodels (Simple Exponential Smoothing)
- **Modeling:** Scikit-learn (Decision Tree, Random Forest, PR-AUC), Matplotlib
- **Environment:** Google Colab / Jupyter
- **Visualization:** Tableau (interactive dashboard, scatter plots, action filters)

---

## Running it

1. Clone the repository.
2. Download the Kaggle dataset and put the four Train CSVs (labels, beneficiary, inpatient, outpatient) in a `data/` folder in the repo root.
3. Install dependencies: `pip install pandas numpy matplotlib scikit-learn statsmodels`
4. Open `Healthcare_Provider_Fraud_Detection_Analysis.ipynb` and run all cells. This builds the provider-level features, runs the forecasting step, trains both models and prints the comparison.
5. The notebook also writes `final_provider_fraud_data.csv`, the provider-level table behind the Tableau dashboard.

---

**Author:** Aritra Pal
**Connect:** https://www.linkedin.com/in/aritrapal20/
