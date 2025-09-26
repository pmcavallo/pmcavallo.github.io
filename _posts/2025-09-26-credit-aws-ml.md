
# Lending Club Credit Risk — AWS ML Showcase (Governance + Cost Control, under $25)

**Goal:** Prove end‑to‑end ML capability on AWS while operating under tight cost and quota constraints. We emphasize **governance, reproducibility, leakage control, and cost‑aware training** using a real credit‑risk problem.

---

## Executive Story

We built a leak‑safe **application‑time** credit default model with **SageMaker XGBoost 1.7‑1** trained on **Managed Spot**. Because quotas/UI hid parts of the console, we **registered the model programmatically** (Model Registry via SDK) and attached a **metrics JSON** in S3. We validated inference from the **model.tar.gz** rather than leaving an endpoint running, keeping **idle cost at $0**.

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

## What to Showcase (and why it matters)

### 1) Data layout & governance
- Curated CSVs in S3 (`data/hpo/train_v_app.csv`, `validation_v_app.csv`).
- Separate `model-metrics/app_metrics.json` used as **ModelQuality** evidence in the registry.

**Screenshot placeholder**  
![S3 data layout](screenshots/01-s3-data.png "Show train_v_app.csv, validation_v_app.csv and model-metrics/app_metrics.json")  
_Explain in caption: proves disciplined storage, clear separation of data and evaluation artifacts._

---

### 2) Leakage-aware feature selection (application-time whitelist)
- We computed **univariate AUC/KS and |corr|**, removed leaky/high‑risk fields, and exported a **whitelist of 15 numeric features** available at application time.
- This is classic governance: features are safe to use before outcomes are known.

**Screenshot placeholder**  
![Feature screening cell](screenshots/02-feature-screening.png "Notebook cell showing per-feature AUC/KS/|corr| and final whitelist")  
_Caption: emphasizes leakage prevention and reproducibility of the whitelist._

---

### 3) Cost-aware SageMaker training on Spot
- Trained with **built-in XGBoost 1.7-1** on **Managed Spot** for ~58% savings.
- No endpoints were left running; batch inference validated locally from the `model.tar.gz`.

**Screenshot placeholder**  
![Training job Spot savings](screenshots/03-training-spot.png "SageMaker training job page showing image= sagemaker-xgboost:1.7-1 and Spot savings")  
_Caption: demonstrates correct image usage and cost control._

---

### 4) Governance via Model Registry (SDK-first)
- Created a **ModelPackageGroup** and an **Approved ModelPackage** via SDK.
- Attached our S3 **metrics JSON** as `ModelQuality.Statistics`.
- UI variability/quota limits were handled gracefully: programmatic control > manual clicks.

**Screenshot placeholder**  
![Studio output with ModelPackage ARN](screenshots/04-studio-arn.png "Cell output printing ModelPackageArn and S3 metrics path")  
_Caption: proves there is a registered package with metrics evidence even when the console menu is hidden._

**Optional screenshot placeholder**  
![Models page referencing package](screenshots/05-model-ref.png "A Model referencing the ModelPackage ARN")  
_Caption: shows the cross-reference from a Model to the registry package._

---

### 5) Honest evaluation & calibration
- Reported **AUC/KS/PR-AUC/F1@Top20%** on holdout validation (n=176).
- Calibrated probabilities; thresholds chosen for **business targeting** (Top20%).

**Screenshot placeholder**  
![Metrics JSON](screenshots/06-metrics-json.png "Open app_metrics.json in S3")  
_Caption: evidence that the same metrics used in the write-up are attached to the package._

---

### 6) Spend governance and teardown
- **AWS Budgets** configured; stayed within a tight cap.
- **Teardown checklist**: no endpoints, no Batch jobs, no warm pools, idle kernels shut down.

**Screenshot placeholder**  
![AWS Budgets](screenshots/07-budgets.png "Budget alert configuration to prove spend governance")  
_Caption: shows proactive cost management._

---

## Results (Validation)

- **AUC:** 0.844  
- **KS:** 0.604  
- **PR-AUC:** 0.617  
- **F1@Top20%:** 0.531  

_Interpretation:_ With strong leakage controls and a small dataset, the calibrated XGBoost is **trustworthy and deployable**. The governance trail (ModelPackage + metrics) makes it review‑ready.

---

## Reproducibility: register the package programmatically

> Works even if your console hides the “Model Registry” menu.

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

**Screenshot placeholder**  
![Execution role policy](screenshots/08-iam-role.png "Execution role and policy summary (no broad wildcards)")  

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

## Roadmap Fit (for interviews & portfolio)

- **AI Expert Roadmap:** Shows rigorous feature governance, calibrated evaluation, and practical deployment paths (batch/serverless).  
- **Layer‑2 Orchestrator Roadmap:** Demonstrates cost controls, registry governance, and SDK automation that survives account constraints.

---

## Screenshot Index (replace placeholders with your images)

1. `screenshots/01-s3-data.png` — S3 layout with `data/hpo/` and `model-metrics/`.  
2. `screenshots/02-feature-screening.png` — Notebook feature screening table + whitelist.  
3. `screenshots/03-training-spot.png` — Training job page (image + Spot savings).  
4. `screenshots/04-studio-arn.png` — Studio cell printing ModelPackageArn + metrics path.  
5. `screenshots/05-model-ref.png` — Model referencing the ModelPackage ARN.  
6. `screenshots/06-metrics-json.png` — S3 `app_metrics.json` open in the console.  
7. `screenshots/07-budgets.png` — AWS Budgets alert configuration.  
8. `screenshots/08-iam-role.png` — Execution role/policy summary.

> Save your images under a `/screenshots/` folder in the repo and keep the filenames above for a drop‑in experience.

---

## Teardown Checklist (cost safety)

- [ ] Endpoints: **none** listed under SageMaker → Inference → Endpoints  
- [ ] Batch transform jobs: **none running**  
- [ ] Notebook kernels: **idle kernels shut down**  
- [ ] Training jobs: completed; **no warm pools**  
- [ ] Budgets: alert **enabled**  
- [ ] S3: keep only minimal artifacts (model.tar.gz, metrics JSON, train/validation data)

> Model Package/Group metadata are near‑free; keep them for governance unless you need to delete.

---

*Prepared for portfolio/README use. Swap the screenshot placeholders with your images and you’re done.*
