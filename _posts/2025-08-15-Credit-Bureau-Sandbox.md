---
layout: post
title: Credit Bureau Sandbox — Governance Gate & Dashboard Hook (AWS + Tableau)
date: 2025-04-18
---

This project demonstrates hands-on work with bureau-style sandbox data, a credit reporting dashboard hook, and an AWS-based governance gate for model productionization. 
The repo demonstrates them in a **lightweight, auditable** way—*without* committing secrets or spinning heavy compute. It gives concrete artifacts, one-liners to reproduce behavior, and clear pointers to files.

---

## What’s included (at a glance)
- **Data contract & ETL scaffold:** Bureau-shaped schema and feature derivations for a sandbox context.
- **Governance gate (SageMaker, no-compute):** Enforces thresholds from a metrics JSON in S3 (AUC ≥ 0.65, KS ≥ 0.20, PSI < 0.10).
- **Audit evidence:** A captured failing run (PSI=0.20) checked into the repo under `CB_Sandbox/evidence/`.
- **Dashboard hook:** README-wired PNG plus a small CSV you can open in Tableau and export a real PNG within minutes.
- **One-liners:** Windows-friendly commands to reproduce the gate and evidence.

---

## Repository structure (key paths)
```
CB_Sandbox/
  evidence/                     # audit logs (includes failing run at PSI=0.20)
  pipelines/sagemaker/          # pipeline gate entrypoint (no-compute)
assets/
  bureau_dashboard.png          # placeholder image (replace with Tableau export)
  dashboard_sample.csv          # tiny aggregate for quick Tableau build
data_contracts/
  bureau_schema.yaml            # schema for bureau-like sandbox data
docs/
  model_card_template.md        # template for documenting variables/risks
  audit_log_schema.json         # sample structure for audit logs
governance/
  policy_map.yaml               # threshold/policy configuration
pyspark_etl_modeling/           # feature derivations & ETL scaffold
README.md                       # thresholds, S3 metrics path, execution ARN
```

---

## How to run (quick start one-liners, Windows)

**Prereqs:** Python 3.10+, AWS CLI configured, S3 bucket writable, and a role to pass to SageMaker.

**1) Recreate a *failing* evaluation (PSI=0.20), upload to S3, and run the gate (logs saved as evidence):**  
```
python -c "import json,os; os.makedirs(r'outputs\metrics',exist_ok=True); json.dump({'auc':0.66,'ks':0.25,'psi':0.20}, open(r'outputs\metrics\evaluation.json','w'))" && aws s3 cp "%CD%\outputs\metrics\evaluation.json" "s3://aws-flagship-project/outputs/metrics/evaluation.json" && python CB_Sandbox\pipelines\sagemaker\pipeline_gate.py --role <YOUR_ROLE_ARN> --bucket s3://aws-flagship-project --prefix cbsandbox --metrics_s3 s3://aws-flagship-project/outputs/metrics/evaluation.json > CB_Sandbox\evidence\gate_failed.txt 2>&1
```

**2) Replace placeholder with a real Tableau PNG:**  
```
# In Tableau: open assets\dashboard_sample.csv, build a couple of views (Delinquency Trend, Segment Mix, Score vs Default), export PNG to assets\bureau_dashboard.png, then:
git add assets/bureau_dashboard.png && git commit -m "Dashboard: Tableau export" && git push
```


---

## Deep dive: 

### 1) **What is a “bureau sandbox” and why it matters**

A *bureau sandbox* is a safe, non-production environment that mimics credit-bureau data—using de-identified or synthetic records aligned to real bureau schemas and distributions. It lets teams prototype ETL, features, dashboards, and governance without handling PII or live customer data.

**What it’s used for**
- **Feature & ETL rehearsal:** derive variables like delinquency counts, utilization, inquiries, thin-file flags, and cohort segments before touching production feeds.
- **Policy & monitoring dry-runs:** test release thresholds (AUC/KS/PSI), drift and stability checks, and adverse-action reason logic.
- **Dashboard development:** build and iterate on credit reporting views (delinquency trend, segment mix, score vs default) without compliance risk.
- **Operational rehearsal:** validate schemas, data quality rules, and pipeline orchestration so the production cutover is low-risk and audit-ready.

**Common fields it simulates**
- **Identity keys (tokenized)** and **trade/collection aggregates** (balances, limits, utilization).
- **Delinquency/charge-off indicators** and **roll-rate histories**.
- **Inquiry counts**, **new-account flags**, **file age/length**.
- **Thin-file** indicators (e.g., ≤2 tradelines and/or short history), which are important for inclusive-credit analysis.

**Limitations & guardrails**
- Distributions are approximate; sandbox data should **not** be used for final model calibration or policy cut decisions without production validation.
- Rare events may be under-represented; treat performance metrics as directional.
- Always document mapping assumptions from sandbox → vendor feeds to avoid drift on go-live.

**How this repo applies the concept**
- We define a bureau-shaped schema in `data_contracts/bureau_schema.yaml`, derive example features in `pyspark_etl_modeling/`, and evaluate stability via PSI/KS/AUC thresholds enforced by an AWS pipeline gate. The dashboard hook (`assets/bureau_dashboard.png` + `assets/dashboard_sample.csv`) enables rapid visualization without exposing sensitive data.

**What’s implemented**
- A reproducible schema (`data_contracts/bureau_schema.yaml`) that mirrors bureau-style fields (IDs, tradeline aggregates, delinquency/charge-off flags, thin-file markers).
- An ETL/feature scaffold (`pyspark_etl_modeling/`) demonstrating safe, sandboxed transformations.
- Drift & stability checks are part of the model evaluation story (PSI, KS, AUC thresholds).


**Quick Verify (no secrets)**
```
python - << "PY"
print("Schema (first 40 lines):")
print("\n".join(open("data_contracts/bureau_schema.yaml","r",encoding="utf-8").read().splitlines()[:40]))
PY
```

---

### 2) **credit reporting dashboard**
**What’s implemented**
- The README image embed is pre-wired to `assets/bureau_dashboard.png` so docs never break.
- A tiny **sample aggregate** CSV (`assets/dashboard_sample.csv`) is provided to drag into Tableau Public and export a real PNG within minutes.

**Suggested first views (Tableau)**
- **Delinquency Trend** (line): monthly delinquency/charge-off rate.
- **Segment Mix** (stacked bar): Prime vs Near-Prime vs Subprime volumes.
- **Score vs Default** (overlay): binned score distribution against default rate.

**Workflow**
1. Open `assets/dashboard_sample.csv` in Tableau Public.  
2. Create 2–3 worksheets as above.  
3. Export as PNG → overwrite `assets/bureau_dashboard.png`.  
4. Commit and push.

---

### 3) **productionizing models in AWS**
**What’s implemented**
- A **no-compute SageMaker pipeline gate** that enforces simple release criteria directly from an S3-hosted metrics JSON: **AUC ≥ 0.65, KS ≥ 0.20, PSI < 0.10**.
- **Audit evidence**: a failing run at PSI=0.20 is checked into `CB_Sandbox/evidence/gate_failed.txt`.
- The README records the **S3 metrics path** and an **execution ARN** for a prior gate run.

**Why this matters**
- Shows how to wire AWS services safely (no secrets committed) and express release governance as code.
- Demonstrates traceability: metrics → policy thresholds → pass/fail + stored evidence.

**Reproduce locally (Windows one-liner)**  
*(Use your role/bucket; the command writes a metrics JSON to S3 and runs the gate)*  
```
python -c "import json,os; os.makedirs(r'outputs\metrics',exist_ok=True); json.dump({'auc':0.66,'ks':0.25,'psi':0.20}, open(r'outputs\metrics\evaluation.json','w'))" && aws s3 cp "%CD%\outputs\metrics\evaluation.json" "s3://aws-flagship-project/outputs/metrics/evaluation.json" && python CB_Sandbox\pipelines\sagemaker\pipeline_gate.py --role <YOUR_ROLE_ARN> --bucket s3://aws-flagship-project --prefix cbsandbox --metrics_s3 s3://aws-flagship-project/outputs/metrics/evaluation.json > CB_Sandbox\evidence\gate_failed.txt 2>&1
```

---

## Outcomes
- **Sandbox data familiarity:** I worked with bureau-shaped schemas and built ETL/feature steps suitable for a sandbox environment, with drift and stability checks aligned to credit risk practice.
- **Dashboard-ready:** I wired the doc to accept a Tableau export and provided a sample aggregate so a visual can be produced immediately—no repo changes required.
- **AWS productionization:** I encoded release thresholds as a SageMaker pipeline gate, captured failing-run evidence, and documented the S3-driven evaluation path. This is how I keep “audit-ready” while moving fast.

---

## What I would add next (if time permits)
- **CI smoke test:** a tiny GitHub Action that imports `pipeline_gate.py` to catch breakage early.
- **Model card + artifacts:** fill in `docs/model_card_template.md` with selections & risks.
- **Real Tableau PNG:** replace the placeholder with an exported `assets/bureau_dashboard.png`.
- **Optional**: Extend ETL to include segmentation and cohort drift monitoring as scheduled jobs.


---

### Output
- **Sandbox data:** schema + ETL + drift checks present and documented.  
- **Dashboard:** image embed is wired; sample CSV enables a real export in minutes.  
- **AWS:** pipeline gate with thresholds, S3-driven metrics, and audit evidence.
