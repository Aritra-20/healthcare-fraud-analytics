# Healthcare Provider Fraud Detection

Medicare fraud happens at the provider level. A hospital or clinic bills for more than it should, usually over and over. The catch is that the data doesn't come one row per provider. It comes one row per claim, and there are over 500K of them.

In this project I turned those claims into features for each provider, trained a class-weighted Random Forest to flag the fraudulent ones, and measured how well it works with PR-AUC.

Notebook: [healthcare-provider-fraud-detection-analysis.ipynb](healthcare-provider-fraud-detection-analysis.ipynb)

Tableau dashboard: https://public.tableau.com/views/Medicare_Fraud_Analytics/Dashboard1

## Results

The model scores a PR-AUC of 0.712 on the test set. A model guessing at random would score about 0.093 (the share of fraudulent providers in the test set), so this is roughly 7.7 times better than chance.

A single Decision Tree trained the same way only got to 0.456. The forest is clearly the better model. The tree is still handy when you need to explain the logic to someone, because you can draw it out as a set of rules.

| Model | PR-AUC |
|---|---|
| Random Forest | 0.712 |
| Decision Tree (depth 5) | 0.456 |
| Random guessing | 0.093 |

### What drives the predictions
| Feature | Importance |
|---|---|
| Total_Reimbursed | 0.336 |
| Total_Deductible | 0.182 |
| Inpatient_Ratio | 0.102 |
| Total_Claims | 0.090 |
| Unique_Patients | 0.083 |
| Avg_Claim_Duration | 0.080 |
| Claims_Per_Patient | 0.064 |
| Avg_Patient_Risk | 0.062 |

The top two are both about money. Fraudulent providers bill far larger totals than the rest. A lot of that is simply size, though, since bigger providers file more claims and bill more. The inpatient ratio coming third makes sense too, as hospital admissions are the most expensive claims to inflate. The more specific behaviour features, like claims per patient and patient risk, helped less than I expected.

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

### Combining the claims
I stacked the inpatient and outpatient claims into one table, with a column marking which type each claim was. Missing payment amounts were set to 0, on the assumption that a blank means nothing was paid.

### Feature engineering
The fraud label is per provider but the data is per claim, so the main job was turning 558K claims into one row per provider. For each patient I first counted how many of the 11 chronic conditions they have, which gives a simple risk score. I also worked out how many days each claim covered.

Then I grouped everything by provider and built these features:

| Feature | What it measures |
|---|---|
| Total_Claims | How many claims the provider filed |
| Unique_Patients | How many different patients they billed for |
| Claims_Per_Patient | Claims divided by patients. Billing the same patients again and again is a common fraud pattern |
| Total_Reimbursed | Total amount Medicare paid them |
| Total_Deductible | Total amount their patients paid out of pocket |
| Avg_Claim_Duration | Average length of a claim in days |
| Inpatient_Ratio | Share of their claims that were hospital admissions |
| Avg_Patient_Risk | How many chronic conditions their patients have on average |

### The model
Only about 9% of providers are fraudulent, so a normal model would learn to say "not fraud" almost every time and still look accurate. I used a Random Forest with `class_weight='balanced'`, which makes missing a fraud case cost the model much more than a false alarm. The data was split 80/20 with `stratify` so the test set keeps the same 9% fraud rate.

### Why PR-AUC and not accuracy
A model that calls every provider "not fraud" would be about 91% accurate and catch nobody. PR-AUC doesn't fall for that. It looks at precision (of the providers flagged, how many really are fraud) and recall (how much of the fraud gets caught) at every possible cut-off, so it only rewards a model that actually finds the rare fraud cases.

### Comparing with a Decision Tree
I trained a single Decision Tree (max depth 5, same class weighting) on the same split and scored it with the same PR-AUC method, to check whether the extra complexity of a forest was worth it.

### Tableau
The provider-level table is exported as `final_provider_fraud_data.csv`, which is what the Tableau dashboard is built on.

## Tools
Python, Pandas, Scikit-learn, Matplotlib, Tableau

## Running it
Open the notebook on Kaggle, add the dataset `rohitrox/healthcare-provider-fraud-detection-analysis` and run all cells. It only uses the four Train files. If you run it somewhere else, change the file paths in the loading cell to wherever you saved the CSVs.

---

Aritra Pal · [LinkedIn](https://www.linkedin.com/in/aritrapal20/)
