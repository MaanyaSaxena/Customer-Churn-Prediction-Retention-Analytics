# KKBox Customer Churn Prediction & Retention Analytics

> SQL (DuckDB) · Python · scikit-learn · statsmodels · SciPy · Tableau

Ranks subscribers by churn risk using historical KKBox subscription data, tests how reliable and how leakage-prone that ranking is, describes which customer segments churn most, and turns the scores into a **scenario-based** retention-targeting policy.

## 1. Business problem

A subscription business can only contact a limited number of customers. Which customers are about to stop renewing, which observed behaviours go with churn, and who is worth contacting first under explicit cost and value assumptions?

## 2. Findings in plain language

- **Churn is concentrated.** The at-risk pool is 12.3% of customers but holds 80.9% of the remaining churners; 38.8% of the pool churns.
- **Auto-renewal is the clearest segment signal.** 57.8% of low-auto-renew customers churn, against 26.2% of high-auto-renew customers (medium effect, Cramér's V = 0.31).
- **Short plans and past cancellations stand out.** Customers on plans of 7 days or less churn at 80.9% (only 1,505 customers); customers who have cancelled before churn at 57.8% against 32.5% for those who never did (odds ratio 2.85).
- **Tenure and spend barely differ.** Both differences are statistically detectable but negligible in size (rank-biserial r of -0.05 and 0.09).
- **Gradient boosting ranks churners better than logistic regression**, and the gap is clear (paired bootstrap, Section 6).
- **Much of the apparent accuracy comes from cancellation, recency and renewal signals close to the label.** Remove them and gradient-boosting ROC-AUC falls from 0.926 to 0.845, and logistic regression from 0.899 to 0.721.
- **Under the stated assumptions**, contacting customers by expected value beats contacting everyone or only the top 10% by risk. That is a modelled estimate, not a measured campaign result.

## 3. Data and design

| Item | Detail |
|---|---|
| Source | Historical KKBox churn data (members, transactions, renewals); raw files are **not** in this repo |
| Size | 992,931 cohort customers; 21,547,746 transactions in the pre-cutoff file (15.8M after keeping cohort customers up to the cutoff and removing invalid rows) |
| Feature cutoff / label cutoff | 28 Feb 2017 / 31 Mar 2017 |
| Target | Official Kaggle churn flag `is_churn_official` |
| Label check | A constructed 30-day renewal label compared with the official flag: recall 0.82, precision 0.82, kappa 0.81 for the chosen rule (cancel-aware expiry, membership-extended renewal), against 0.74 / 0.63 / 0.66 for the original max-expiry rule. It is never used as a feature. |
| Model table | 987,567 customers, one row each (5,364 set aside: already churned before the cutoff or no history) |
| At-risk pool (modelled) | 121,280 customers with an observable outcome; split 90,960 train / 30,320 test, stratified |
| Features | 47 for gradient boosting; 46 for logistic regression (`total_amount_paid` dropped for VIF > 10) |

The pool differs sharply from other customers (lapsed, auto-renew off, recent cancellations), so the model describes this pool, not every subscriber.

## 4. Pipeline

```text
Raw CSVs -> DuckDB staging + quality checks -> label construction -> customer-level features
   -> model table -> logistic regression + gradient boosting + baselines -> held-out evaluation
   -> leakage sensitivity -> bootstrap CIs and paired comparison -> segment hypothesis tests
   -> value-based retention scenarios -> aggregate CSV exports -> Tableau
```

Features use only transactions on or before the cutoff (asserted in the notebook). Scaling and imputation statistics come from the training split only.

## 5. Why these methods

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Interpretable model | Logistic regression (odds ratios per +1 SD) | Tree only | Benchmark to beat and readable effects |
| Non-linear check | Gradient boosting | Linear only | Tests whether non-linearity adds value |
| Metrics | ROC-AUC, PR-AUC, precision/recall/lift @ top 10% | Accuracy | 38.8% churn rate and a limited contact budget make ranking metrics the relevant ones |
| Uncertainty | Bootstrap CIs; paired bootstrap for the model gap | Separate intervals | Two marginal intervals cannot show whether the gap is real |
| Odds ratios under the conservative feature set | L2 logistic + stratified bootstrap | Wald p-values | Stable with correlated indicators; no Wald p-values for penalised fits |
| Categorical vs churn | Chi-square + Cramér's V (Fisher exact if a 2x2 table has expected counts < 5) | p-value only | Effect size matters more than significance at this n. Expected counts were all above 500, so chi-square was valid. |
| Continuous vs churn | Welch t-test if both groups roughly symmetric, else Mann-Whitney U | Student's t | Tenure and spend were both skewed (skew 1.3 to 3.6), so Mann-Whitney was used; variances were similar (Levene p > 0.4) |
| Multiple tests | Benjamini-Hochberg and Holm across all six tests | Bonferroni | Bonferroni is overly conservative |

## 6. Results

**Held-out model comparison** (30,320 customers, churn prevalence 38.8%):

| Model | ROC-AUC | PR-AUC | Precision @ top 10% | Recall @ top 10% | Lift @ top 10% |
|---|---:|---:|---:|---:|---:|
| Rule: last transaction was a cancel | 0.6449 | 0.5490 | 0.9086* | 0.2343* | 2.34* |
| Auto-renew ratio only | 0.6752 | 0.5195 | 0.5834 | 0.1504 | 1.50 |
| Logistic regression | 0.8993 | 0.8189 | 0.9172 | 0.2365 | 2.36 |
| Gradient boosting | **0.9255** | **0.8784** | **0.9809** | **0.2529** | **2.53** |

\*The rule is a 0/1 flag, so its top-10% depends on tie-breaking; judge it by AUC and by the policy table.

**Calibration.** Logistic-regression predicted and observed churn agree within about 4 percentage points in every risk decile; the model slightly over-predicts in the top decile (0.956 vs 0.917) and under-predicts in the middle (0.289 vs 0.326 in decile 5).

**Uncertainty (95% bootstrap CIs, 2,000 replicates).**

| Metric | Logistic regression | Gradient boosting | Difference (GBM - LR) |
|---|---|---|---|
| ROC-AUC | 0.8993 (0.8959-0.9025) | 0.9255 (0.9229-0.9283) | +0.026 (0.024-0.028) |
| PR-AUC | 0.8189 (0.8113-0.8268) | 0.8784 (0.8736-0.8839) | +0.059 (0.054-0.065) |
| Precision @ 10% | 0.9172 (0.9060-0.9261) | 0.9809 (0.9753-0.9858) | +0.064 (0.055-0.075) |

All difference intervals exclude zero. They condition on the fitted models and do not capture refitting or temporal-shift uncertainty.

**Leakage and target-proximity sensitivity** (same split; logistic rows are an L2 refit):

| Feature set | Logistic ROC-AUC | Logistic PR-AUC | GBM ROC-AUC | GBM PR-AUC | GBM precision @ 10% |
|---|---:|---:|---:|---:|---:|
| Original | 0.8993 | 0.8192 | 0.9255 | 0.8784 | 0.981 |
| Drop `last_is_cancel` | 0.8814 | 0.7741 | 0.9234 | 0.8743 | 0.977 |
| Conservative (also drops recency, cancellation counts, auto-renew and plan-length signals) | 0.7206 | 0.6124 | 0.8451 | 0.7686 | 0.902 |

The conservative set is a stress test, not proof that the remaining features are leakage-free. `last_is_cancel` has an odds ratio in the hundreds (about 357 per +1 SD in this run, 95% CI 294-432). Its exact value has varied between runs (roughly 300 to 400) because the flag almost separates the classes, so report it as "very large" rather than a precise figure. The flag also helps define the cancel-aware expiry used in label construction.

**Segment hypothesis tests** (whole at-risk pool, n = 121,280; six tests, adjusted p-values < 0.001 for all):

| Variable | Test | Effect size | Churn pattern |
|---|---|---|---|
| Auto-renew group | Chi-square (χ² = 11,890, df 2) | V = 0.313 (medium) | low 57.8%, medium 40.5%, high 26.2% |
| Plan group | Chi-square (χ² = 3,812, df 3) | V = 0.177 (small) | ≤7d 80.9%, 8-30d 35.0%, 31-90d 50.1%, >90d 54.7% |
| Tenure group | Chi-square (χ² = 658, df 3) | V = 0.074 (negligible) | <6m 47.9%, 6-12m 33.7%, 1-2y 38.4%, >2y 38.1% |
| Cancel history | Chi-square (χ² = 6,126, df 1) | V = 0.225 (small) | cancelled before 57.8% vs never 32.5% (OR 2.85, 95% CI 2.78-2.93) |
| Tenure (days) | Mann-Whitney U | r = -0.051 (negligible) | median 740 churned vs 772 retained (difference -32 days, 95% CI -50 to -17) |
| Spend per 30 days | Mann-Whitney U | r = 0.091 (negligible) | median 144.0 vs 137.7 NTD (difference +6.3, 95% CI 5.8 to 6.7) |

With this many customers almost any difference is significant, so read the effect sizes. These are observational associations, and marginal differences can differ from the adjusted effects in the regression.

**Retention economics** (30,320 test customers; assumed NTD 30 contact cost, 20% success rate, 6-month value horizon):

| Policy | Contacted | Precision | Recall | Est. revenue saved | Contact cost | Est. net gain |
|---|---:|---:|---:|---:|---:|---:|
| Contact nobody | 0 | - | 0.000 | 0 | 0 | 0 |
| Whole at-risk pool | 30,320 | 0.388 | 1.000 | 1,901,020 | 909,600 | 991,420 |
| Last transaction was a cancel | 4,010 | 0.908 | 0.310 | 546,448 | 120,300 | 426,148 |
| Top 10% by risk | 3,032 | 0.917 | 0.236 | 397,362 | 90,960 | 306,402 |
| Expected gain > 0 (value-based) | 16,890 | 0.652 | 0.937 | 1,830,650 | 506,700 | **1,323,950** |

Net gain stays positive in all 16 cost and success-rate combinations tested (NTD 71,248 to 3,571,418), but its size depends heavily on the assumptions. The break-even churn probability falls with customer value (0.25 for low-value, 0.18 for mid-value, 0.16 for high-value customers), so high-value customers are worth contacting at lower risk. These are scenario estimates, not campaign ROI.

## 7. Recommendations

1. Use the score to prioritise outreach, with human review, not as an automatic action.
2. Validate when each feature is actually available at prediction time and how close it is to the label before any live use.
3. Prioritise low-auto-renew and short-plan customers in any pilot; they show the largest observed churn gaps.
4. Run a randomised holdout to measure incremental retention before trusting the economics.
5. Monitor precision, recall, contact rate and net contribution by segment, and revalidate on recent data (this dataset is from 2017).

## 8. Limitations

- Historical 2017 data from one service; results may not transfer.
- The official label and the constructed renewal check use different operational definitions.
- Cancellation, recency and renewal features are close to the label and can overstate prospective performance.
- Retention value, contact cost and success rate are assumptions, not measurements.
- Hypothesis tests and odds ratios describe associations, not causes.
- Not done yet: power analysis for a retention experiment, formal calibration statistics beyond the decile table, an independent replication in another tool.

## 9. Reproduce

```bash
pip install -r requirements.txt
export KKBOX_DATA_DIR=/path/to/kaggle/csvs      # train.csv, transactions.csv, transactions_v2.csv, members_v3.csv
export KKBOX_WORK_DIR=.                          # optional
export COHORT_SAMPLE_PCT=10                      # optional quick trial run (use 100 for the full results)
jupyter notebook Notebook.ipynb                  # run all cells top to bottom
```

The raw data come from the KKBox churn competition on Kaggle and are not redistributed here. Seeds are fixed, rows are ordered by customer id, and ties among same-day transactions are broken explicitly (cancellation first), identically in the label and feature SQL. The figures above come from a fresh end-to-end run of this version, in which the label step, the model table and the modelling code all report the same counts (987,567 customers; 121,280 in the pool).

## 10. Repository structure and outputs

```text
├── README.md
├── Notebook.ipynb
├── requirements.txt
├── sql/                        # written by the notebook: 01_load ... 06_model_table
├── dashboard_exports_public/   # approved aggregate CSVs only
└── diagrams/                   # dashboard screenshots
```

The notebook writes these aggregate files to `dashboard_exports/`: `data_quality_audit`, `label_quality`, `odds_ratios`, `odds_ratios_with_95ci`, `model_comparison`, `calibration_by_decile`, `feature_sensitivity`, `bootstrap_metric_confidence_intervals`, `paired_model_comparison_bootstrap`, `conservative_logistic_odds_ratios_bootstrap`, `hypothesis_tests`, `segment_churn_rates`, `assumption_checks`, `policy_comparison`, `threshold_by_value_tier`, `sensitivity_net_gain` (all `.csv`), and `results_summary.json`. A customer-level priority table containing `msno` exists only in memory and must never be exported. There is no automated privacy audit, so check every file for customer ids before publishing.

## 11. Tableau dashboards

Built from the aggregate exports; refresh the figures shown on the dashboards so they match the tables above.

1. **Executive Churn Overview** - model comparison and headline KPIs. [Live dashboard](https://public.tableau.com/app/profile/maanya.saxena/viz/CustomerChurnAnalysis_17924841151790/ExecutiveChurnOverview)
   ![Executive Churn Overview](diagrams/executive-churn-overview.png)
2. **Churn Drivers & Retention Strategy** - drivers and retention-policy economics (modelled estimates). [Live dashboard](https://public.tableau.com/app/profile/maanya.saxena/viz/SurvivalAnalysis_17924939558530/ChurnDriversRetentionStrategy?publish=yes)
   ![Churn Drivers and Retention Strategy](diagrams/churn-drivers-retention-strategy.png)

## 12. Skills demonstrated

SQL data preparation and label validation · feature engineering without look-ahead · binary classification and benchmarking · ROC-AUC/PR-AUC/lift evaluation · bootstrap and paired-bootstrap inference · hypothesis testing with assumption checks and effect sizes · leakage and target-proximity analysis · scenario economics · privacy-aware aggregate reporting · communicating assumptions and uncertainty
