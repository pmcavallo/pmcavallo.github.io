---
layout: post
title: AI-in-the-Cloud Knowledge Shootout
date: 2025-09-14
---
Perplexity vs NotebookLM, as a continuation of the Cross-Cloud Shootout series. The project's goal is to evaluate whether Perplexity and NotebookLM can operate as cloud knowledge orchestrators, producing reliable, grounded guidance on cloud architecture, cost optimization, and governance.

## Controlled Prompts
Keep wording identical across tools.

Categories:
- Cost Optimization
- Architecture Fit (Telecom latency scenario)
- Governance and Compliance

There are 2 steps for each category, meaning two different questions and prompts within each category to a total of 6 steps in this project.

### Why the Corpus Was Different

NotebookLM requires an explicit, curated corpus. For each prompt I fed it a defined set of sources (AWS/GCP docs, pricing pages, my Cross-Cloud project). This makes it behave like a **research assistant**: it sticks tightly to those inputs and won’t improvise beyond them.

Perplexity, by contrast, does not let us preload a corpus. It always searches the open web in real time, then cites what it finds. That makes it more like a **field guide**: broad coverage, but with less alignment to our specific setup.

#### Implications
- **Comparability** — Results are not perfectly “apples to apples.” NotebookLM is bounded by what we feed it, while Perplexity can range wider.
- **Bias** — NotebookLM reflected my Cross-Cloud project so strongly that it sometimes echoed my language back. Perplexity surfaced perspectives we hadn’t included, sometimes fresher but less grounded.
- **Use Cases** — This divergence is exactly the point of the shootout: one tool excels when you want controlled, source-aligned synthesis; the other when you want quick breadth and up-to-date tactics.

---

## Step 1: COST_01 Prompt (Budget-Constrained ML Pipeline)

### What We’re Seeing
- **NotebookLM**  
  - Very detailed, structured response (general principles → AWS pipeline → GCP pipeline → comparison).  
  - Grounded heavily in your Cross-Cloud Shootout blog and official framework docs.  
  - Rich with cost ranges, risks, trade-offs, and even AutoML details.  
  - Reads like a *mini whitepaper* and feels influenced by your prior work.  

- **Perplexity**  
  - Much shorter, in bullet-point style.  
  - Pulls from blogs, cost-optimization guides, and Google Cloud docs.  
  - Provides service lists, pricing ranges, and tactics — but at a higher level, less context-specific.  
  - More of a *cheat sheet* than a detailed strategy.  

### Interpretation
This difference is exactly why this shootout matters:  

- **NotebookLM** = Stronger when you want **depth, synthesis, and alignment to your curated corpus**. It feels like a research assistant.  
- **Perplexity** = Stronger when you want **quick, broad, up-to-date pointers** across the open web. It feels like a briefing tool.  

For **COST_01**, NotebookLM is “better” if you need detailed architecture and cost planning. Perplexity is “better” if you just want to get the highlights quickly.  

---

## Step 2: ARCH_01 — Telecom-Grade Latency Architecture

### What We’re Seeing
- **NotebookLM**  
  - Detailed, structured description of architectures for both AWS and GCP.  
  - For AWS: emphasized Local Zones, Wavelength Zones, CloudFront, SageMaker endpoints, CloudFront/ElastiCache caching, multi-AZ/region failover, and observability via CloudWatch.  
  - For GCP: emphasized region distribution, global private network, Cloud Load Balancing, Vertex AI Endpoints, Cloud CDN/Memorystore, multi-zone failover, and Cloud Monitoring/Trace/Logging.  
  - Reads like a comprehensive *solution design document* — step-by-step through each architectural layer.  


- **Perplexity**  
  - Shorter and more *bullet-style* again:contentReference[oaicite:1]{index=1}.  
  - For AWS: highlighted Wavelength Zones, edge-first deployment, SageMaker/EKS at edge and core, ElastiCache, multi-AZ and Route 53 for failover, CloudWatch for observability.  
  - For GCP: focused on global load balancer, Direct Interconnect, Vertex AI endpoints, Memorystore/CDN caching, multi-zone clusters, Cloud Operations stack for observability.  
  - Included a neat *component comparison table* mapping AWS vs GCP across networking, placement, serving, caching, failover, and observability.  

### Interpretation
- **NotebookLM** gave a textbook-style deep dive — clear, linear, and exhaustive, drawing heavily on your curated corpus. It’s almost an *architectural blueprint*.  
- **Perplexity** was more concise, emphasizing edge deployments, 5G/Wavelength specifics, and presented the info in a way that’s faster to skim, with a useful side-by-side comparison table.  
- The contrast is consistent with Step 1: NotebookLM = depth and thoroughness; Perplexity = agility and comparative framing. For latency-heavy telecom use cases, NotebookLM shows *how* to build it, while Perplexity shows *what components* to choose.  


---

## Step 3: GOV_01 — Compliance Frameworks Comparison (AWS vs GCP)

### What We’re Seeing
- **NotebookLM**  
  - Gives a structured, policy-first comparison of **IAM, policy enforcement, data residency, and auditability** for AWS vs GCP, then ties them to **SOC 2 / GDPR** responsibilities. Emphasizes AWS Control Tower + Organizations (SCPs) and GCP Organization Policy (managed/list/custom constraints, dry-run), with clear examples like `constraints/gcp.resourceLocations` for residency and Cloud Audit Logs / Policy Intelligence for auditing. Reads like a governance playbook grounded in official mechanisms. 


- **Perplexity**  
  - Covers the same dimensions but in a **shorter, skimmable** style with a summary table (IAM, policy, residency, auditability, responsibilities). Highlights AWS CloudTrail/Config/Artifact and GCP Audit Logs/SCC/DLP, plus high-level SOC 2/GDPR mapping. Pulls broadly from web sources and feels like a quick briefing rather than a deep policy guide. 


### Interpretation
- **NotebookLM** delivers a **governance blueprint**: precise levers (SCPs, Control Tower controls, Org Policy constraints, dry-run) and how they operationalize **SOC 2/GDPR**—excellent when you need exact controls to implement. 
- **Perplexity** gives a **fast comparison** with concise tables and pointers—useful for orientation and stakeholder readouts, but with less procedural detail than the NotebookLM write-up. 

---

## Step 4: COST_02 — Training vs Inference Economics

### What We’re Seeing
- **NotebookLM**  
  - Provides a structured walkthrough of cost design for ML training and inference across AWS and GCP.  
  - For **AWS**, emphasizes SageMaker Autopilot, Spot Instances, right-sizing endpoints, lifecycle policies, and pay-per-second billing as core tactics. Gives concrete ranges (e.g., Autopilot ~$10 per 30 min run) and highlights transparent billing.  
  - For **GCP**, centers on Vertex AI and BigQuery ML, showing cost trade-offs and risks: strong unified platform, but with “hidden quota gates” and potential budget friction. BigQuery ML flagged as a pragmatic, cost-controlled middle ground.  
  - Ends with a clear comparison table contrasting AWS vs GCP in philosophy, AutoML experience, training, in-database ML, predictability, and cost-control levers:contentReference[oaicite:0]{index=0}.


- **Perplexity**  
  - Summarizes AWS vs GCP pricing in a quick, tabular style with concrete hourly examples (e.g., SageMaker ml.m5.xlarge $0.23/hr, Vertex AI n1-standard-4 $0.15/hr).  
  - Covers both training and inference, with assumptions for a mid-size team (5–7 data scientists, CPU-heavy workloads with some GPU).  
  - Lists optimization levers: AWS (Savings Plans, Spot, rightsizing, idle shutdowns, S3 lifecycle) vs GCP (Spot VMs, Committed Use Discounts, endpoint co-hosting, tagging/monitoring).  
  - Provides a side-by-side cost structure table for CPU/GPU training, inference, and storage:contentReference[oaicite:1]{index=1}.


### Interpretation
- **NotebookLM** delivers a **playbook-style cost analysis** — strong on governance of spend, with detailed tactics and a decision lens (AWS = transparency, GCP = unified but quota-friction). Great for teams planning policy and budget strategy.  
- **Perplexity** offers a **snapshot with numbers** — concrete hourly costs and practical optimization tactics in a single table. Better suited for quick estimations and briefings.  
- Together, NotebookLM is **deep and prescriptive**, while Perplexity is **fast and pragmatic**.  

---

## Step 5: ARCH_02 — Hybrid Multi-Cloud ML Control Plane

### What We’re Seeing
- **NotebookLM**  
  - Frames the architecture through the **Well-Architected principles**, extending from Step 2’s design.  
  - For **AWS**, emphasizes SageMaker AI, S3, Glue/EMR, SageMaker Pipelines, and Step Functions for orchestration, with detailed monitoring (Model Registry, Model Monitoring).  
  - For **GCP**, focuses on Vertex AI, BigQuery ML, Dataflow/Dataproc, Vertex Pipelines, Model Registry, Explainable AI, and Model Monitoring.  
  - Highlights **trade-offs**: AWS = transparent billing + mature ecosystem but complex integration; GCP = unified platform but hidden quota gates and risk of operational friction.  
  - Positions BigQuery ML as a pragmatic middle ground for structured data:contentReference.

- **Perplexity**  
  - Produces a **low-latency telecom-focused hybrid design** spanning AWS and GCP.  
  - AWS side: Wavelength Zones, Outposts, EKS/SageMaker for edge + regional inference, ElastiCache for caching, Route 53 for failover, CloudWatch/X-Ray for observability.  
  - GCP side: Global Load Balancer, Interconnect, Vertex AI endpoints, Memorystore/CDN for caching, multi-regional VMs with autoscaling, Stackdriver for observability.  
  - Provides a **component comparison table** (Networking, Placement, Model Serving, Caching, Failover, Observability).  
  - Emphasizes risks like vendor lock-in, data egress fees, and budget spikes if not controlled:contentReference.


### Interpretation
- **NotebookLM** approaches the question as a **governance-driven blueprint**: aligning AWS/GCP service choices with Well-Architected principles, orchestration, monitoring, and cost-control, but staying closer to general ML pipeline design.  
- **Perplexity** delivers a **concrete hybrid architecture for telecom-style use cases**, complete with edge/latency optimizations, failover, and observability specifics.  
- In effect, NotebookLM is stronger on **conceptual orchestration and governance**, while Perplexity shines with **practical system-level architecture diagrams and edge considerations**.  

---

## Step 6: GOV_02 — Model Governance Baseline

### What We’re Seeing
- **NotebookLM**  
  - Provides a **structured, baseline governance framework** spanning AWS and GCP.  
  - Anchored in Well-Architected and Responsible AI principles.  
  - Outlines **documentation, approvals, monitoring, lineage, reproducibility**, and links these directly to AWS tools (SageMaker Model Cards, Model Registry, Audit Manager) and GCP equivalents (Vertex ML Metadata, Responsible AI, Model Monitoring).  
  - Reads like a **policy playbook**, heavily grounded in the official frameworks and your Cross-Cloud context.
 

- **Perplexity**  
  - Produces a **practical governance checklist** with emphasis on PD model needs.  
  - Details specific **risks of poor governance** (bias, data drift, lack of reproducibility).  
  - Strong on **service mappings**: AWS (SageMaker Model Cards, Audit Manager, Control Tower) vs GCP (Vertex AI Metadata, Responsible AI, Assured Workloads).  
  - Framed as a **set of actionable steps** for mid-size enterprises, rather than a full baseline policy.


### Interpretation
- **NotebookLM** = stronger for defining a **formal baseline policy framework** — if you were drafting documentation to hand to compliance officers or governance teams.  
- **Perplexity** = stronger for **actionable implementation guidance** — if you’re in a data science team that needs concrete steps and mappings to tools.  
- In essence, NotebookLM “thinks like a policy author,” while Perplexity “thinks like a practitioner writing a runbook.”

---

## Aggregated Results (Steps 1–6)

| Tool        | Prompt   | Grounding | Completeness | Depth | Clarity | Reproducibility* | Weighted Score |
|-------------|----------|-----------|--------------|-------|---------|------------------|----------------|
| Perplexity  | COST_01  |    3      |      3       |   2   |    4    |        –         |      2.95      |
| NotebookLM  | COST_01  |    5      |      5       |   5   |    4    |        –         |      4.70      |
| Perplexity  | ARCH_01  |    4      |      4       |   3   |    5    |        –         |      3.85      |
| NotebookLM  | ARCH_01  |    5      |      5       |   5   |    4    |        –         |      4.70      |
| Perplexity  | GOV_01   |    4      |      4       |   3   |    5    |        –         |      3.85      |
| NotebookLM  | GOV_01   |    5      |      5       |   5   |    4    |        –         |      4.70      |
| Perplexity  | COST_02  |    4      |      4       |   3   |    5    |        –         |      3.85      |
| NotebookLM  | COST_02  |    5      |      5       |   5   |    4    |        –         |      4.70      |
| Perplexity  | ARCH_02  |    4      |      4       |   4   |    5    |        –         |      4.10      |
| NotebookLM  | ARCH_02  |    5      |      5       |   5   |    4    |        –         |      4.70      |
| Perplexity  | GOV_02   |    4      |      4       |   3   |    5    |        –         |      3.85      |
| NotebookLM  | GOV_02   |    5      |      5       |   5   |    4    |        –         |      4.70      |

\*Reproducibility not yet scored — will require repeat runs.

### How to Read the Scores

Each tool was scored across five dimensions:

- **Grounding (30%)** — How well the answer is tied to the curated corpus (docs, blogs, official references) versus generic web output.  
- **Completeness (20%)** — Whether all parts of the prompt were addressed, with no major gaps.  
- **Depth (20%)** — The richness of reasoning, trade-offs, and context beyond surface-level lists.  
- **Clarity (15%)** — How well-structured and easy to understand the answer is (e.g., tables, flow, brevity).  
- **Reproducibility (15%)** — Consistency of results if the same prompt were run again (not yet tested in this shootout).  

Weighted scores are normalized to exclude reproducibility (since it wasn’t tested), so NotebookLM’s consistent 4.70 reflects near-max performance across the other four categories.

### Dimension Highlights

- **Grounding** — NotebookLM excelled here, consistently tying its outputs to the curated corpus (Cross-Cloud blog + cloud docs). Perplexity was grounded too, but leaned more on broad web sources.  
- **Completeness** — NotebookLM almost always covered every part of the prompt; Perplexity sometimes skipped deeper trade-offs or risks.  
- **Depth** — NotebookLM produced blueprint- and policy-level reasoning, while Perplexity stayed surface-level but pragmatic.  
- **Clarity** — Perplexity’s tables and bullet lists made it the clearest tool for quick reading. NotebookLM was structured but verbose.  
- **Reproducibility** — Not tested in this shootout; both tools will need repeat runs to evaluate stability.  

Notes on Weighted Score:
- Weighting scheme: Grounding (30%), Completeness (20%), Depth (20%), Clarity (15%), Reproducibility (15%).
- Since reproducibility wasn’t tested, scores were normalized over 0.85.
- That’s why NotebookLM consistently lands at 4.70 (near ceiling), while Perplexity ranges from ~2.95 to ~4.10.

## Conclusion
This shootout shows that **NotebookLM** and **Perplexity** occupy complementary roles. NotebookLM excels when you need **deep synthesis, structured reasoning, and policy-level alignment** to curated sources. Perplexity shines when you need **concise, actionable insights and comparative tables** drawn from the broader web.  

Together, they illustrate a future where LLMs can help orchestrate cloud strategy:  
- NotebookLM as the **research assistant** that drafts frameworks and governance playbooks.  
- Perplexity as the **field guide** that offers quick cost checks, architecture comparisons, and implementation tactics.  

Neither is a “winner” outright; the real strength comes from pairing them. The combination highlights how AI can support both **strategic policy formation** and **day-to-day operational decision-making** in cloud computing.
