---
layout: post
title: RiskBench AI Coding Shootout (Claude Code, Cursor, Github Copilot)
date: 2025-09-14
---

This project set out to pit three leading AI coding assistants (GitHub Copilot, Claude Code, and Cursor) against each other in a controlled “shootout,” with each tool tasked to build out the same end-to-end machine learning pipeline. Across four sprints, the tools generated synthetic datasets, trained and tuned XGBoost models, explored data quality and feature engineering, and ultimately deployed a serving API with SHAP-based interpretability. By holding the repo, prompts, and acceptance tests constant, the project revealed not just raw coding differences, but how each tool shapes data quality, model credibility, and the path to a production-ready ML system

---

# Suggestions to Stay Comparable

- Use same prompts for each tool → apples-to-apples.
- Use full file prompts when possible → conserve requests.
- Always run the same tests/static checks → uniform evaluation.
- Keep logs/screenshots/prompt history → traceability.

---

# RiskBench Shootout Tools

This shootout now includes **three tools** used on the **same repo** with the **same prompts and tests**:

- GitHub Copilot (inline IDE)
- Cursor (inline IDE)
- Claude Code (terminal/agent)

Evaluation remains identical across tools (tests, metrics, artifacts). Claude Code may edit files and run commands, but success is judged by the same local pytest/static checks.

# Splitting the Project in Chunks/Sprints

I decided to split the project in 4 sprints to make sure I was keeping rich notes between stages, and confirm each tool was doing eveything as planned before the final product.

Sprint 1: Set up the repo structure, dependencies, and a working **data generator** with acceptance tests. 
Sprint 2: Focused on training a baseline XGBoost model across the three tools.
Sprint 3: Expose the trained model via API and add lightweight interpretability.
Sprint 4: Build a production-style serving layer with FastAPI/Uvicorn that delivers predictions and SHAP-based explanations through stable endpoints

## 1. Sprint 1 
The purpose of Sprint 1 was to test how three AI-assisted coding tools, **GitHub Copilot**, **Claude Code**, and **Cursor**, perform when tasked with generating the initial RiskBench package structure, producing synthetic training data, and running the provided tests.

This sprint serves as a baseline to evaluate:
- **Setup smoothness**: How easily each tool got running.  
- **Data generation**: Ability to produce a valid `train.parquet`.  
- **Test execution**: Reliability and completeness of unit tests.  
- **Failure modes**: Nature of the issues when something went wrong.  

---

## 2. What Worked
- **All three tools successfully generated the `riskbench` package scaffolding** (pyproject, CLI, modules, tests).  
- **Data generation worked across the board**: Copilot and Claude produced a `train.parquet` file with the expected structure and size.  
- **Copilot completed both data generation and tests cleanly** with minor environment adjustments.  

---

## 3. What Didn’t Work
- **Claude**: Several tests failed despite data generation succeeding. Failures were mainly due to assertion mismatches (`positive_rate` range) and unsupported aggregation (`mean`) on categorical dtypes.  
- **Cursor**: Data generation crashed due to Pandas attempting to cast `MonthEnd` objects into `int64`. As a result, no `train.parquet` was created in `C:\riskbench\data`. To proceed fairly, we copied the **Copilot-generated dataset** into Cursor’s data folder. This preserves Cursor’s failure as evidence while allowing Sprint 2 comparability.  
- **Copilot**: Needed more manual setup than the others (virtual environment, ensuring pyproject install worked), but once configured, all tests passed.  

---

## 4. Tool Comparisons
- **GitHub Copilot**
  - **Strengths**: Clean execution of tests, reliable pipeline once setup was fixed.  
  - **Weaknesses**: Setup friction (manual environment work required).  

- **Claude Code**
  - **Strengths**: Very smooth initial structure generation, good CLI handling.  
  - **Weaknesses**: Logical/semantic issues in test validation (e.g., rate range, aggregation errors).  

- **Cursor**
  - **Strengths**: Fast initial code generation, minimal friction to produce structure.  
  - **Weaknesses**: Data generation failed (`MonthEnd` casting bug), leaving dataset empty. Required manual dataset copy from Copilot to continue.  

---

## 5. Sprint 1 Takeaways
- Having **three tools side by side** reveals not only coding ability but also **error patterns**:  
  - Copilot leans on clean, conventional outputs but needs more manual direction.  
  - Claude excels at scaffolding but struggles with edge-case correctness.  
  - Cursor generates aggressively but has brittleness in handling types.  
- For the shootout, **failures are not setbacks but valuable evidence** of how each tool handles real-world complexity.  
- Cursor’s dataset failure is especially important: it highlights that **baseline functionality (writing to disk)** cannot be taken for granted, even when scaffolding looks correct.  

---

# Sprint 2 Summary: Baseline XGBoost Training

## Overview
Sprint 2 focused on training a baseline XGBoost model across the three tools. 
Initial results revealed severe **data leakage** (perfect scores), which was then corrected by excluding `leakage_col` and `timestamp`. 
The fixed runs highlighted divergences in performance across tools.

---

## Results by Tool (Fixed Implementations)

![claude](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/ccode4.png?raw=true)

### Claude (Control)
- **Train AUC**: 0.9510  
- **Test AUC**: 0.8220  
- **Train Accuracy**: 0.9123  
- **Test Accuracy**: 0.8427  
- **KS Statistic**: 0.4885  

Claude delivered a **credible baseline model**, with strong class separation and expected generalization.  
This became the **control reference** for comparison.

---

### Copilot
- **Train AUC**: 0.9375  
- **Test AUC**: 0.4929  
- **Train Accuracy**: 0.9108  
- **Test Accuracy**: 0.9087  
- **KS Statistic**: 0.0288  

Copilot’s implementation produced **random-like test performance**:  
- High accuracy is misleading due to **class imbalance** (majority class prediction).  
- Very low KS and AUC indicate **poor discriminatory power**.  
- Indicates either **feature weakness** or collapse of predictive signal after leakage removal.

---

### Cursor
- **Train AUC**: 0.8051  
- **Test AUC**: 0.5002  
- **Train Accuracy**: 0.9087  
- **Test Accuracy**: 0.9087  
- **KS Statistic**: 0.0207  

Cursor’s implementation mirrored Copilot’s:  
- Essentially random performance (Test AUC ≈ 0.50).  
- KS near zero confirms **no class separation**.  
- Accuracy inflated by **majority-class prediction**.  
- Cursor flagged **feature engineering needs** and dataset limitations.

![cursor](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/cursor5.png?raw=true)

---

## Divergence Analysis
- **Claude retained signal** → Test AUC ≈ 0.82, credible baseline.  
- **Copilot and Cursor collapsed to random performance** → Test AUC ≈ 0.49–0.50, KS ≈ 0.02–0.03.  
- Both Copilot and Cursor flagged issues:
  - Class imbalance (≈9% positive rate).  
  - Weak synthetic features after leakage exclusion.  
  - Need for stronger feature engineering to recover predictive signal.

---

## Decision
✅ **Claude’s model is adopted as the baseline control** going forward.  
- Provides stable, credible results.  
- Establishes a common benchmark for Sprint 3 and beyond.  

⚠️ **Copilot and Cursor’s degraded models are preserved as evidence**.  
- Document important divergences.  
- Highlight orchestration challenges and tool differences.  
- Serve as a reminder of why orchestration (not blind execution) matters.

---

## Key Learnings
1. Leakage prevention works, but exposes dataset weaknesses.  
2. Class imbalance skews accuracy; AUC and KS are more reliable.  
3. Claude’s implementation showed the dataset still carries predictive signal.  
4. Copilot and Cursor revealed the **true difficulty of the task** without engineered features.  
5. Future work must emphasize **feature engineering, regularization, and imbalance handling**.

---

# Sprint 3 Re-run on Unified Dataset

## Context
Following Sprint 3.5, we unified all tools onto **Claude’s dataset** (`train.parquet`) to ensure fair comparability.  
Goal: re-run Sprint 3 (baseline + tuned modeling) for Copilot and Cursor using Claude’s data.

---

## Claude (Control)
- Already trained and tuned on its own dataset (Claude’s data).  
- Baseline AUC ≈ 0.822, Tuned AUC ≈ 0.840, KS ≈ 0.52【50†source】.  
- These results serve as the control for unified comparisons.

---

## Copilot (Re-run with Claude Data)
- Dataset: Claude’s unified `train.parquet`  
- Baseline metrics:  
  - ROC AUC: **0.8188**  
  - KS: **0.4824**  
  - Accuracy: **0.8392**  
  - Confusion matrix: [[11450, 640], [1772, 1138]]  

- Tuned metrics:  
  - ROC AUC: **0.8387**  
  - KS: **0.5163**  
  - Accuracy: **0.8487**  
  - Confusion matrix: [[11645, 445], [1825, 1085]]  

- Best hyperparameters:  
  - colsample_bytree=0.8, learning_rate=0.1, max_depth=3, min_child_weight=3,  
    n_estimators=200, reg_alpha=0.1, reg_lambda=1, subsample=0.8  


**Interpretation**: Copilot’s tuned results (AUC 0.839, KS 0.516) are nearly identical to Claude’s (AUC 0.840, KS 0.520).  
This confirms the original divergence was due to **dataset quality**, not modeling code.

---

## Cursor (Re-run with Claude Data)
- Attempted re-run blocked by **Cursor usage limits** (free plan quota reached).  
- Current status: **postponed until quota reset or Pro upgrade**.  
- Evidence retained:  
  - Original Sprint 3 results on Copilot’s dataset (AUC ≈ 0.50, KS ≈ 0).  
  - Sprint 3.5 diagnostics confirmed Cursor and Copilot shared the same weaker dataset.  
- Decision: Copilot’s re-run serves as proxy evidence for dataset effect. Cursor re-run marked pending.

---

## Decision
- Continue Sprint 4 (serving + interpretability) with **Claude and Copilot (Claude’s dataset)**.  
- Document Cursor’s blocked status transparently in RiskBench log.  
- Preserve original datasets (`train_original.parquet`) for reproducibility.

---

# Sprint 3.5 – Data Understanding & Feature Engineering

## Objective
Diagnose why Claude produced credible models (AUC ~0.82) while Copilot and Cursor flatlined near random performance.  
Focus: dataset quality, feature signal, class imbalance, and feature engineering opportunities.

![cursor](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/cursor8.png?raw=true)

---

## Results by Tool

### Claude
- **Positive rate**: 19.4% (scale_pos_weight ≈ 4.15)  
- **Top univariate feature AUC**: `credit_grade` = 0.6934  
- **Other strong predictors**: `credit_score` (0.6546), `debt_to_income` (0.6141)  
- **Signal**: Clear hierarchy of predictive features, meaningful KS values  
- **Correlations**: Moderate, interpretable  
- **Feature Engineering Suggestions**: 12 (interactions, ratios, binning, temporal features)  
- **Narrative**: Dataset has strong predictive signal; imbalance manageable.

---

### Cursor
- **Positive rate**: 9.13% (scale_pos_weight ≈ 9.96)  
- **Top univariate feature AUC**: `cat_1` = 0.5095 (barely above random)  
- **Signal**: Near-random across all features; maximum KS = 0.0048  
- **Correlations**: Extremely weak (<0.01)  
- **Missingness**: num_3 (33%), num_5 (37%)  
- **Feature Engineering Suggestions**: 12 (target encoding, temporal aggregations, ratios, missing flags)  
- **Narrative**: Dataset weak; almost no inherent predictive signal; heavy reliance on FE.

---

### Copilot
- **Positive rate**: 9.13% (scale_pos_weight ≈ 9.96)  
- **Top univariate feature AUC**: `cat_1` = 0.5095  
- **Signal**: Near-random; features ~0.50 AUROC, very low KS  
- **Correlations**: Extremely weak (|r| < 0.01)  
- **Feature Engineering Suggestions**: 10 (winsorization, binning, encodings, ratios, clustering)  
- **Narrative**: Same dataset as Cursor; weak raw signal; emphasizes engineered features to compensate.

---

## Strategic Takeaways
- **Claude’s dataset (Sprint 1)** is fundamentally richer: more positives, stronger predictors.  
- **Copilot & Cursor datasets** (shared) are weaker: ~9% positives, near-random features.  
- Divergence is due to **dataset generation quality**, not model training differences.  
- Feature engineering is critical for Copilot/Cursor, while Claude’s models already benefit from strong raw signal.

---

## Execution Friction & Developer Effort
- **Copilot**: Last to start but first to finish. Requires minimal clicks/approvals → fastest workflow.  
- **Claude**: Produces the richest insights but requires frequent approvals, slower overall.  
- **Cursor**: Similar interaction overhead as Claude but without comparable payoff.  

---

## Next Steps
1. Define a **common feature engineering spec** (using best ideas from Claude, Cursor, Copilot).  
2. Re-run models with engineered features to normalize dataset advantage.  
3. Proceed to **Sprint 4**: serving (FastAPI endpoints) + interpretability (SHAP).  

---

# Project Note: Dataset vs EDA/Feature Engineering

## Context
Sprint 3.5 revealed a fundamental reason for divergence between tools:
- **Claude’s models performed well (AUC ~0.82 baseline, ~0.84 tuned).**
- **Copilot and Cursor collapsed to random-like predictions (AUC ~0.50).**

## Key Findings
1. **Dataset Generation (Sprint 1)**
   - Claude generated a synthetic dataset with:
     - ~19% positive rate (vs. ~9% for Copilot/Cursor)
     - Stronger feature–target relationships
     - Realistic correlations between predictors
   - Copilot’s dataset was weaker, and Cursor inherited Copilot’s directly.

2. **EDA and Feature Engineering**
   - Claude often performs implicit EDA/feature preparation steps (e.g., checking distributions, encoding, correlation checks) even before being explicitly prompted.
   - Copilot and Cursor tend to follow the literal prompt only, without enriching the workflow.

3. **Evidence from Sprint 3.5**
   - **Claude** surfaced a clear signal hierarchy (credit_grade, credit_score, debt_to_income).  
   - **Copilot/Cursor** showed features barely above random (best AUROC ~0.51), very weak correlations, and severe imbalance (~10:1).  
   - Their feature engineering suggestions were generic “manufacture signal” strategies, while Claude proposed targeted, high-value transformations.

## Interpretation
Claude’s advantage comes from **both factors combined**:
- It generated a **better synthetic dataset** in Sprint 1.  
- It also tends to perform **deeper implicit analysis** (EDA/FE) without being told.  

By contrast, Copilot and Cursor stuck to literal execution and were constrained by weaker data.

## Decision Implication
To ensure fairness and isolate tool performance in Sprints 4+, I will **unify all tools onto Claude’s dataset** (`train.parquet`).  
- This ensures the comparison measures coding/orchestration ability, not dataset luck.  
- Sprint 1 divergence will remain documented as evidence of tool differences in data generation.

---

# Sprint 4: Serving & Explainability Report

**Environment:** Windows / PowerShell (`py` launcher), FastAPI + Uvicorn

## What went wrong (root causes)

### Copilot
- **Wrote files outside** its directory, breaking provenance and making it unclear which artifacts were in use.
- **Placeholder artifacts** (e.g., dummy pickles) caused `pickle`/`joblib` load failures.
- Result: repeated crashes when trying to start an API with Copilot’s outputs.

### ChatGPT (assistant guidance missteps)
- Proposed patches that **weren’t always single‑line PowerShell edits** (against my rule).
- Introduced a `/meta` file that **referenced undefined globals** or **nonexistent model attributes**, producing 500s.
- A regex edit left a **dangling block**, causing `SyntaxError` at import.

### Serialization under Uvicorn
- Some preprocessors had been serialized under `uvicorn.__main__`, so the class path didn’t resolve at runtime.

---

## How Claude fixed it

- Restored a clean **`serving.py`**:
  - Single, robust `/meta` using explicit `MODEL_NAME="xgboost"`, `MODEL_VERSION="S3_tuned"`, and `FEAT_NAMES` (14 features).
  - Safe extraction of SHAP base value (optional).
  - Removed duplicate/partial endpoints and stray code.
- Ensured **preprocessor loading** uses `joblib.load(path)` and is resilient to module aliasing.
- Validated the app via **Swagger** and curl/irm POSTs; all endpoints returned 200.

**Net effect:** API consistently boots, serves predictions and SHAP explanations, and exposes stable metadata and schema endpoints for screenshots.

![claude](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/ccode13.png?raw=true)

---

## Interpretation of your test result

```json
{"model":"xgboost","version":"S3_tuned","features":["credit_score","annual_income","debt_to_income","credit_utilization","credit_history_length","recent_inquiries","open_accounts","age","employment_status","housing_status","loan_purpose","state","education_level","credit_grade"],"threshold":0.5,"base_value":null}
```

![riskbench](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/riskbench.png?raw=true)


```json
{
  "score": 0.2540644705295563,
  "model": "xgboost",
  "version": "S3_tuned",
  "reasons": [
    {"feature":"credit_grade","value":"A","shap_value": 2.368015766143799},
    {"feature":"credit_utilization","value":0,"shap_value":-0.69468092918396},
    {"feature":"credit_history_length","value":0,"shap_value": 0.6206353902816772},
    {"feature":"credit_score","value":700,"shap_value":-0.5636231899261475},
    {"feature":"recent_inquiries","value":0,"shap_value":-0.5194531083106995}
  ]
}
```

- **Score = 0.254** → with the default threshold **0.5**, this is a **negative class** decision.
- **SHAP contributions** (log‑odds):
  - Positive → pushes risk/probability **up**; negative → pushes it **down**.
  - `credit_grade = "A"` contributed the largest **increase** (+2.37).
  - `credit_utilization = 0`, `credit_score = 700`, and `recent_inquiries = 0` **decreased** the score (risk‑reducing).
- Business mapping: if the positive class is “high risk,” negatives are protective; if it’s “approval,” flip the interpretation. Either way, the reasons clearly justify a ~0.25 score.

---

## Final Takeaway

Claude Code consistently outperformed by generating stronger data, richer feature engineering, and delivering a fully functional serving API, while Copilot proved fastest with minimal interaction but weaker at deeper analysis, and Cursor was limited by quota and dataset weaknesses. This shows that tool differences aren’t just about speed, they shape data quality, model credibility, and whether you end up with a real, production-ready ML system.

Overall, I came away impressed with what these coding agents can already do: in just a few sprints, they scaffolded a repo, generated data, trained and tuned models, and stood up a serving layer with interpretability.The experience highlighted both their speed and their blind spots, showing that while automation can push projects forward quickly, the real value comes from guiding, refining, and connecting those outputs into something robust and strategically useful.


 
