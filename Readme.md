# Healthcare Provider Fraud Detection & Analytics

> **Live Interactive Dashboard:** https://public.tableau.com/views/Medicare_Fraud_Analytics/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

## 📌 Executive Summary
Healthcare fraud costs the Medicare system billions of dollars annually. The challenge in detecting this fraud is that raw data exists at the *individual claim* level, while fraud investigations and patterns occur at the *provider* level. 

This project transforms over 500,000 raw Medicare inpatient and outpatient claims into a provider-level risk dataset. By engineering behavioral and financial features, I trained a machine learning model to identify providers with high-risk billing patterns, successfully navigating a heavy class imbalance (~9.4% fraud rate).

---

## 📊 Key Business Insights
Based on the Random Forest feature importance extraction and visual analysis in Tableau, fraudulent providers separate themselves from benign providers through three distinct anomalies:

1. **Overtreatment Indicators:** High average claims per unique beneficiary. Fraudulent providers consistently bill multiple claims per patient at rates far exceeding benign providers.
2. **Financial Scale:** Unusually high total reimbursement amounts compared to peer averages.
3. **Targeted Demographics:** A tendency to treat beneficiaries with a higher average "Chronic Condition Burden" (patient risk score), suggesting potential upcoding or the targeted exploitation of highly vulnerable, high-morbidity demographics.

---

## 🛠️ Data & Methodology

### 1. Data Source
The dataset utilizes de-identified Medicare provider data, consisting of four relational tables: Inpatient Claims, Outpatient Claims, Beneficiary Details, and Provider Fraud Labels. 
* **Source:** [Kaggle - Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

### 2. Feature Engineering (Claim-to-Provider Aggregation)
To transition from claim-level records to provider-level labels, I engineered several custom features:

| Feature Name | Description | Business Rationale |
| :--- | :--- | :--- |
| **Claims_Per_Bene** | Ratio of total claims to unique beneficiaries. | Flags providers who repeatedly bill for the same patients. |
| **Patient_Risk_Score** | Sum of chronic conditions per patient, averaged by provider. | Identifies providers targeting high-morbidity patients for higher billing. |
| **Avg_Claim_Duration** | Average length of inpatient/outpatient care. | Highlights anomalous treatment timelines and extended stays. |

### 3. Machine Learning Modeling
* **Algorithm:** Random Forest Classifier.
* **Handling Imbalance:** The training data contained a heavily imbalanced 90.6% / 9.4% split of benign to fraudulent providers. Instead of artificially oversampling (e.g., SMOTE) which can introduce synthetic noise, I utilized `class_weight='balanced'` to heavily penalize the model for missing minority class instances.
* **Evaluation Metric:** Accuracy is highly misleading in imbalanced datasets (a model guessing "Benign" every time would be ~91% accurate). I evaluated the model using **Precision-Recall AUC (PR-AUC)**, achieving a score of **0.713**. This demonstrates a strong ability to isolate true fraud cases without overwhelming auditors with false positives.

---

## 💻 Tech Stack
* **Data Manipulation:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, PR-AUC evaluation)
* **Environment:** Google Colab / Jupyter Notebook
* **Visualization:** Tableau (Interactive Dashboards, Scatter Plots, Custom Action Filters)

---

## 🚀 How to Run the Code
1. Clone the repository to your local machine.
2. Download the Kaggle dataset and place the 8 CSV files (Train and Test splits) in a `data/` folder in your root directory.
3. Open the `Provider_Fraud_Analytics.ipynb` notebook.
4. Run all cells to execute the data aggregation, feature engineering, and model training pipeline.

---

**Author:** Aritra Pal  
**Connect:** https://www.linkedin.com/in/aritrapal20/