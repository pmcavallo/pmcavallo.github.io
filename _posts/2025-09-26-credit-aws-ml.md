---
layout: post
title: Lending Club Credit Risk — AWS ML Showcase (Governance + Cost Control, under $25)
date: 2025-03-26
---

This project demonstrates a **budget‑conscious, console‑first ML pipeline** on AWS: data profiling with **AWS Glue DataBrew**, feature curation and storage in **Amazon S3**, training and packaging **XGBoost** in **Amazon SageMaker** with **Managed Spot Training**, registering the model in the **SageMaker Model Registry**, and offline scoring/metrics suitable for a **batch decisioning** use case. Guardrails include **Budgets**, **Anomaly Detection**, and deletion/stop procedures to keep spend near **$0/hr** when idle.

---

## Executive Story

I built a leak‑safe **application‑time** credit default model with **SageMaker XGBoost 1.7‑1** trained on **Managed Spot**. Because quotas/UI hid parts of the console, we **registered the model programmatically** (Model Registry via SDK) and attached a **metrics JSON** in S3. We validated inference from the **model.tar.gz** rather than leaving an endpoint running, keeping **idle cost at $0**.

**Validation metrics (n=176):** `AUC ≈ 0.844`, `KS ≈ 0.604`, `PR-AUC ≈ 0.617`, `F1@Top20% ≈ 0.531`  
**Training billable time:** ~57 s (135 s total) — **~58% Spot savings**  
**Artifacts:** model tarball in S3 + metrics JSON + Approved ModelPackage (ARN)  

This project demonstrates **mastery of AWS and ML**: thoughtful **feature governance**, **cost management**, **SDK‑first reliability** when the UI gets in the way, and clean **teardown**.

---

## Architecture (minimal, production‑minded)

```
S3 (raw → curated)              SageMaker Studio (Notebook)
   │                                   │
   ├── data/hpo/train_v_app.csv ──────▶│ Feature screening (AUC/KS/|corr|) → whitelist (15 numeric features)
   ├── data/hpo/validation_v_app.csv   │
   │                                   │
   └── model-metrics/app_metrics.json◀─┘ Exported evaluation (AUC, KS, PR-AUC, F1@Top20)

SageMaker Training (Managed Spot, built-in XGBoost 1.7-1)
   └── output/model.tar.gz → S3

SageMaker Model Registry (SDK)
   └── ModelPackageGroup: credit-risk-xgb-app
       └── Version 1 (Approved) with ModelQuality metrics
```

**Why this design?** It shows cost‑aware training, **governance (leakage control + registry)**, and reproducibility **without** always‑on endpoints.

---

## 1) Problem & Objective

**Business question**  
Predict the probability that a Lending Club loan will default (“target_bad” proxy). Outputs are used for **pricing / cutoffs** and **portfolio monitoring**.

**Constraints**  
- Console‑first (no heavy infra templates), **small budget**.  
- **Governed**: metrics and artifacts must be traceable, **no always‑on endpoints**.  
- **Interpretable** enough for credit stakeholders.

**Outcome**  
A lean, reproducible pipeline and a packaged model with documented metrics (AUC, PR‑AUC, KS, and an operating point like F1@Top20%).


## 2) Architecture at a glance

- **Amazon S3** — raw files, curated training/validation, batch outputs, and model metrics JSON.  
- **AWS Glue DataBrew** — data **profiling / validation** and quick visual exploration.  
- **Amazon SageMaker Studio** — one‑click notebooks (for training & packaging), **Managed Spot** for cost savings.  
- **SageMaker Model Registry** — model package + metrics for governance; **no live endpoint** required.  
- **CloudWatch Logs** — retention set to 7 days.  
- **AWS Budgets & Cost Anomaly Detection** — alerts if spend drifts.

> **Cost note**: With no endpoints/apps running, the environment sits at **$0/hr**; residual is S3/Logs pennies.

---

## 3) Data & EDA with AWS Glue DataBrew

I used **DataBrew** to profile a 1K-row sample for fast, visual checks before training.
**Dataset**: 1000 rows × 151 columns sample from Lending Club “accepted” dataset for demo (full workflow scales to millions).

### 3.1 DataBrew grid & profiling

### Grid preview (sanity check)
![DataBrew grid – quick scan of columns](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/lend.png?raw=true)

Key takeaways:
- 151 columns with a mix of numeric and categorical features.
- Missingness concentrated in a subset of fields (DataBrew highlights valid vs. missing cells).

### Profiling summary (rows, columns, nulls, duplicates)
![DataBrew profile summary panel](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/databrew2.png?raw=true)

**Data types:** ~100 numeric, ~50 categorical. **Valid cells:** ~72% in sample; **0 duplicate rows**.

### 3.2 Correlations & value distributions
### Correlations heatmap (leakage sniff test)
Pairs such as `loan_amnt` vs. `funded_amnt` and `installment` highlight redundancy—useful for feature pruning.
![DataBrew correlations heatmap](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/databrew3.png?raw=true)

What I used this for:
- Identify **leakage candidates** (e.g., `funded_amnt` vs `loan_amnt`, or post‑outcome features).  
- Sanity‑check **scale** and outliers prior to model training.

### Distribution comparison across numeric fields
Helpful to spot skew/outliers before tree-based modeling.
![DataBrew distribution comparison boxplots](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/databrew4.png?raw=true)

### 3.3 Column‑level summary
### Column-level quality & distinctness
![Columns summary](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/databrew5.png?raw=true)

**Why DataBrew:** It speeds up first‑pass quality checks without code, and it leaves a **profile job artifact** you can save alongside the model lineage.

---

## 4) Baseline model (Scikit-learn Logistic Regression)

Before turning on SageMaker training, I built a quick baseline to set an honest yardstick.

![Baseline logistic metrics and confusion matrix](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/logit.png?raw=true)

- AUC ≈ **0.789** on the validation slice  
- Strong precision on the majority class, expected weak recall on the minority class

---

## 5) SageMaker XGBoost (managed, Spot-optimized)

We trained XGBoost with leakage-safe features, tracked metrics (**AUC, PR-AUC, KS, F1@Top20%**), and saved predictions for auditability.

![SageMaker XGBoost metrics + code snippet](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/xgboost.png?raw=true)

**Highlights**

- **Spot Training** achieved ~**58–65%** cost savings for short jobs  
- Clean train/validation splits persisted to S3 for reproducibility  
- Post-train, we extracted `model.tar.gz`, scored the validation set offline, and wrote both **predictions** and **metrics** back to S3

---

## 6) Packaging & (attempted) registry entry

We demonstrated programmatic packaging for a **model package** and a **package group**—useful for enterprises with a Model Registry workflow. UI access is account/region-feature-dependent, so we show **code-first registration** and expose artifacts.

- Model artifact & image selection in the console 
![SageMaker model create view](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/model.png?raw=true)

- Code-first package group + package creation (SDK), ARNs captured for governance
![Model package group/ARNs evidence](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/registry.png?raw=true)

> If the account has the Model Registry UX hidden, the SDK path still records package groups and ARNs; otherwise, fall back to **Models → Create model** and **Batch transform** for deployment and scoring.

---

## 7) Repro: Step-by-step

1. **Upload & sample** data to S3 under `data/raw/`  
2. **DataBrew profile** the sample (export HTML if desired)  
3. **Create train/validation** CSVs into `data/hpo/`  
4. **Train** SageMaker XGBoost (Spot enabled), writing artifacts to `output/calibration_*`  
5. **Score** in-notebook for this footprint; for scale, use **Batch Transform** with the saved model  
6. **(Optional)** Register a model package programmatically; otherwise create a **Model** and attach to an **Endpoint**/**Batch Transform** job when quotas allow  
7. **Publish** the repo with these screenshots and metrics for stakeholders

---

## 8) Cost controls & teardown

- **Controls used:** Budgets, Spot training, no endpoints left running, notebooks stopped after use  
- **End-of-day checklist**
  - Terminate/stop Studio kernels and images  
  - Ensure **Endpoints = 0**, **Training jobs = Completed**, **Processing = none running**  
  - Leave S3 data/artifacts for the portfolio; they cost cents at this scale

---

## 9) Results & lessons

- **Comparable AUC** to the baseline with more flexible ranking metrics (**PR-AUC, KS, F1@Top20%**)  
- **DataBrew** accelerated validation and leakage checks without local wrangling  
- **SageMaker** gave packaging and artifact discipline even when full registry UI wasn’t present  
- **Governance:** Everything is S3-versioned and reproducible with code + console trails

---


## What to Showcase (and why it matters)

### Data layout & governance
- Curated CSVs in S3 (`data/hpo/train_v_app.csv`, `validation_v_app.csv`).
- Separate `model-metrics/app_metrics.json` used as **ModelQuality** evidence in the registry.

**Screenshot placeholder**  
![s3](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/s3.png?raw=true)
_Explain in caption: proves disciplined storage, clear separation of data and evaluation artifacts._

---

### Leakage-aware feature selection (application-time whitelist)
- We computed **univariate AUC/KS and |corr|**, removed leaky/high‑risk fields, and exported a **whitelist of 15 numeric features** available at application time.
- This is classic governance: features are safe to use before outcomes are known.

---

### Cost-aware SageMaker training on Spot
- Trained with **built-in XGBoost 1.7-1** on **Managed Spot** for ~58% savings.
- No endpoints were left running; batch inference validated locally from the `model.tar.gz`.

---

### Governance via Model Registry (SDK-first)
- Created a **ModelPackageGroup** and an **Approved ModelPackage** via SDK.
- Attached the S3 **metrics JSON** as `ModelQuality.Statistics`.
- UI variability/quota limits were handled gracefully: programmatic control > manual clicks.


---

### Honest evaluation & calibration
- Reported **AUC/KS/PR-AUC/F1@Top20%** on holdout validation (n=176).
- Calibrated probabilities; thresholds chosen for **business targeting** (Top20%).

---

### Spend governance and teardown
- **AWS Budgets** configured; stayed within a tight cap.
- **Teardown checklist**: no endpoints, no Batch jobs, no warm pools, idle kernels shut down.

---

## Results (Validation)

- **AUC:** 0.844  
- **KS:** 0.604  
- **PR-AUC:** 0.617  
- **F1@Top20%:** 0.531  

## Reproducibility: register the package programmatically

> Works even if the console hides the “Model Registry” menu.

```python
import boto3, sagemaker
from sagemaker import image_uris

region = sagemaker.Session().boto_region_name
bucket = "credit-risk-flagship-dev"
mpg    = "credit-risk-xgb-app"
image  = image_uris.retrieve("xgboost", region=region, version="1.7-1")

artifact_s3 = "s3://credit-risk-flagship-dev/output/calibration_app/sagemaker-xgboost-2025-09-22-22-54-33-406/output/model.tar.gz"
metrics_s3  = "s3://credit-risk-flagship-dev/model-metrics/app_metrics.json"

sm = boto3.client("sagemaker")
try:
    sm.create_model_package_group(
        ModelPackageGroupName=mpg,
        ModelPackageGroupDescription="App-safe 15-feature XGBoost; leakage-controlled; Spot-trained."
    )
except sm.exceptions.ValidationException:
    pass

resp = sm.create_model_package(
    ModelPackageGroupName=mpg,
    ModelApprovalStatus="Approved",
    ModelPackageDescription="Application-time feature whitelist; validation AUC≈0.844, KS≈0.604.",
    InferenceSpecification={
        "Containers":[{"Image": image, "ModelDataUrl": artifact_s3}],
        "SupportedContentTypes":["text/csv"],
        "SupportedResponseMIMETypes":["text/csv","application/json"]
    },
    ModelMetrics={
        "ModelQuality":{"Statistics":{"ContentType":"application/json","S3Uri": metrics_s3}}
    }
)
print("MODEL PACKAGE ARN:", resp["ModelPackageArn"])
```

**Describe it later (proof):**
```python
import boto3
sm=boto3.client("sagemaker")
arn="arn:aws:sagemaker:us-east-1:678804053923:model-package/credit-risk-xgb-app/1"  # replace if re-created
desc=sm.describe_model_package(ModelPackageName=arn)
print(desc["ModelApprovalStatus"])
print(desc["InferenceSpecification"]["Containers"][0]["Image"])
print(desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"])
print(desc.get("ModelMetrics",{}))
```

---

## Cost & Governance Controls (at-a-glance)

- **IAM:** single execution role; least privilege + Registry API access.  
- **Managed Spot training:** ~58% savings; smallest practical instance.  
- **Zero idle cost:** no endpoints left up; batch validated from tarball.  
- **Budgets:** alert configured; stayed inside a ~$25 envelope.  
- **Tags:** package/group tagged for Project/Stage/Governance/Cost.  

---

## Key AWS Lessons (turning constraints into strengths)

- **UI variability & quotas** are common in real accounts; **SDK‑first** workflows keep you moving and leave a better audit trail.  
- **Leakage control** is not optional in credit risk; feature **availability timing** is part of MLOps governance.  
- **Costs accumulate at the edges** (idle endpoints, forgotten jobs). A teardown checklist is part of “done.”

---

## Next Steps (low-cost polish)

- Add a **Serverless Inference** one‑shot demo (invoke once, screenshot latency, delete).  
- Wrap training+registry into a single `train_register.py` and wire to **CI**.  
- Add a monthly **drift/KS check** as a SageMaker Processing job (trigger on demand).  

---


