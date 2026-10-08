# Healthcare Provider Fraud Detection

Medicare fraud happens at the provider level. A hospital or clinic bills for more than it should, usually over and over. The catch is that the data doesn't come one row per provider. It comes one row per claim, and there are over 500K of them.

In this project I turned those claims into features for each provider, added a forecasting signal for billing that drifts away from a provider's own trend, trained a class-weighted Random Forest to flag the fraudulent ones, and measured how well it works with PR-AUC.

Notebook: [healthcare-provider-fraud-detection-analysis.ipynb](healthcare-provider-fraud-detection-analysis.ipynb) · [Run it on Kaggle](https://www.kaggle.com/code/aritrapal3196/healthcare-provider-fraud-detection-analysis)

Tableau dashboard: https://public.tableau.com/views/Medicare_Fraud_Analytics/Dashboard1

## Results

The Random Forest scores a **PR-AUC of 0.706** on the held-out test set. Guessing at random would score about 0.093 (the share of fraudulent providers in the test set), so the model does about **7.6 times better than chance**.

A single Decision Tree trained the same way only got to 0.464, so the forest is the clearly better model overall.

| Model | PR-AUC |
|---|---|
| Random Forest | **0.706** |
| Decision Tree (depth 5) | 0.464 |
| Random guessing | 0.093 |

### What the two models actually do differently

PR-AUC says the forest is better, but the confusion matrices show a trade-off that matters for how you'd use it. The test set has 1,082 providers, 101 of them fraudulent.

| | Random Forest | Decision Tree |
|---|---|---|
| Fraud caught (recall) | 51 of 101 (50%) | 89 of 101 (88%) |
| Fraud missed | 50 | 12 |
| Honest providers wrongly flagged | 17 | 154 |
| Precision (of those flagged, how many are fraud) | 75% | 37% |

- **The forest is the cautious investigator.** Three out of four providers it flags really are fraudulent, but at the default cut-off it misses half the fraud.
- **The tree casts a wide net.** It catches almost 9 in 10 fraud cases but buries them among 154 false alarms.

In practice, I'd use the forest's fraud *probabilities* rather than its yes/no answer, and set the cut-off to match how many cases the audit team can review. Lowering it would catch more fraud at the cost of more false alarms.

### What drives the predictions

| Feature | Importance |
|---|---|
| Total_Reimbursed | 0.373 |
| Total_Deductible | 0.197 |
| Inpatient_Ratio | 0.075 |
| Unique_Beneficiaries | 0.075 |
| Avg_Claim_Duration | 0.066 |
| Total_Claims | 0.061 |
| Claims_Per_Bene | 0.053 |
| Avg_Patient_Risk | 0.050 |
| Billing_Trend_Deviation | 0.049 |

The top two are both about money, and together they carry about 57% of the model's decisions. Fraudulent providers bill far larger totals than the rest. A lot of that is simply size, though, since bigger providers file more claims and bill more. The inpatient ratio coming next makes sense, as hospital admissions are the most expensive claims to inflate.

The forecasting feature (Billing_Trend_Deviation) ranked last. One month's jump against a provider's own trend turned out to be a weak signal next to sheer billing volume. That's still useful to know: in this data, *how much* a provider bills says more than *how suddenly* it changes.

## The data

This is the Healthcare Provider Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis). It's de-identified Medicare data split across four files:

| File | What's in it | Rows |
|---|---|---|
| Labels | Each provider and whether it was flagged as potential fraud | 5,410 providers |
| Beneficiary | Patient details, including 11 chronic condition flags | 138,556 patients |
| Inpatient claims | Claims where the patient was admitted to hospital | 40,474 |
| Outpatient claims | Claims where the patient wasn't admitted | 517,737 |

That's 558,211 claims in total. Only 506 of the 5,410 providers (9.4%) are labelled as fraudulent.

## How I did it

### 1. Combining the claims
I stacked the inpatient and outpatient claims into one table, with a column marking which type each claim was. Missing payment amounts were set to 0, on the assumption that a blank means nothing was paid.

### 2. Patient and claim features
For each patient I counted how many of the 11 chronic conditions they have, which gives a simple risk score. I also worked out how many days each claim covered, counting same-day claims as 1 day.

### 3. Rolling up to one row per provider
The fraud label is per provider but the data is per claim, so the main job was turning 558K claims into one row per provider:

| Feature | What it measures |
|---|---|
| Total_Claims | How many claims the provider filed |
| Unique_Beneficiaries | How many different patients they billed for |
| Claims_Per_Bene | Claims divided by patients. Billing the same patients again and again is a common fraud pattern |
| Total_Reimbursed | Total amount Medicare paid them |
| Total_Deductible | Total amount their patients paid out of pocket |
| Avg_Claim_Duration | Average length of a claim in days |
| Inpatient_Ratio | Share of their claims that were hospital admissions |
| Avg_Patient_Risk | How many chronic conditions their patients have on average |
| Billing_Trend_Deviation | How far the provider's latest month of billing moved away from its own forecast (see below) |

### 4. Billing-trend forecasting
The other features are a snapshot. I wanted one that captures *change*, so I built a monthly reimbursement series for each of the 5,410 providers. For every provider with at least 4 months of history, I fitted **Simple Exponential Smoothing** (Statsmodels) on all but the last month, forecast that last month, and measured how far the actual figure was from the forecast, as a share of the forecast. A positive number means the provider billed more than its own trend predicted. Providers with too little history got the median value, so they weren't dropped and weren't marked as unusual either.

### 5. The model
Only about 9% of providers are fraudulent, so a normal model would learn to say "not fraud" almost every time and still look accurate. I used a Random Forest (100 trees) with `class_weight='balanced'`, which makes missing a fraud case cost the model much more than a false alarm. The data was split 80/20 with `stratify` so the test set keeps the same 9% fraud rate.

### 6. Why PR-AUC and not accuracy
A model that calls every provider "not fraud" would be about 91% accurate and catch nobody. PR-AUC doesn't fall for that. It looks at precision (of the providers flagged, how many really are fraud) and recall (how much of the fraud gets caught) at every possible cut-off, so it only rewards a model that actually finds the rare fraud cases.

### 7. Comparing with a Decision Tree
I trained a single Decision Tree (max depth 5, same class weighting) on the same split and scored it the same way, to check whether the extra complexity of a forest was worth it. It was, but the tree is still handy when you need to explain the logic to someone, because you can draw it out as a set of rules.

### 8. Tableau
The provider-level table is exported as `final_provider_fraud_data.csv`, which is what the Tableau dashboard is built on.

## Tools
Python, Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Tableau

## Running it
Open the notebook on Kaggle, add the dataset `rohitrox/healthcare-provider-fraud-detection-analysis` and run all cells. It only uses the four Train files. If you run it somewhere else, change the file paths in the loading cell to wherever you saved the CSVs.

---

Aritra Pal · [LinkedIn](https://www.linkedin.com/in/aritrapal20/) · [Portfolio](https://aritrapal.me)
