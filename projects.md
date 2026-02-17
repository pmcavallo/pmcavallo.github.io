---
layout: default
title: Projects
permalink: /projects/
---

A selection of hands-on projects focused on building production AI systems for regulated industries, where auditability, compliance, and source fidelity are not optional. The work spans AI governance tooling, multi-agent orchestration, LLM fine-tuning and alignment, RAG architectures with hallucination prevention, and full MLOps deployment pipelines. Built with Python, Claude API, LangChain, ChromaDB, PyTorch, Streamlit, and AWS, with regulatory grounding in NIST AI RMF, SR 11-7, and FHFA AB 2022-02.

---

**PolicyLens: AI Risk Assessment Engine for Financial Services**  
An AI-powered governance tool that operationalizes four regulatory frameworks — NIST AI 100-1, NIST AI 600-1, FHFA AB 2022-02, and SR 11-7 — into structured compliance assessments. Describe an AI use case, and PolicyLens produces a Risk Committee-ready assessment organized by the NIST AI RMF's four functions (Govern, Map, Measure, Manage), with regulatory citations, internal policy alignment scoring, and a gap analysis identifying where institutional governance falls short of regulatory expectations. Sample assessment of a credit decisioning model revealed **39% average coverage** across RMF functions, with 7 Critical and 1 High severity gaps.

**Highlights:**

* **Three-layer compliance model:** Assesses External Regulation ("What does regulation require?"), Internal Policy ("What does our governance cover?"), and Specific Use Case ("Does this deployment comply?") — then cross-references all three layers to surface gaps
* **Section-aware RAG:** Regulatory documents are parsed preserving hierarchical structure (not naive fixed-window chunking), so every retrieval carries its full section lineage for precise citations
* **Dual-collection retrieval:** Queries run against both regulatory frameworks (251 chunks from 4 documents) and internal policy collections (45 chunks) simultaneously, enabling cross-reference gap analysis
* **Structured output validation:** Every assessment is Pydantic-validated: four RMF functions present, all findings carry citations, severity enums enforced, coverage scores computed
* **Risk Committee report export:** Generates markdown reports in the format a second-line risk team would present to a governance committee, with remediation priorities sorted by severity

**Business / Research Impact:**  
PolicyLens automates the workflow a second-line AI risk team runs manually: assess each AI deployment against external regulatory requirements, cross-reference against internal governance documentation, identify where institutional policy falls short, and produce remediation recommendations with specific citations. It turns weeks of manual policy mapping into a structured, repeatable, auditable process.

**Tech Stack:**

* **LLM / API:** Claude API (claude-sonnet-4-5)
* **Vector Store:** ChromaDB (dual collections, cosine similarity)
* **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
* **PDF Parsing:** PyMuPDF (section-aware extraction with 6 document-specific parsers)
* **Output Validation:** Pydantic schemas
* **UI:** Streamlit (tabbed dashboard with coverage scores, gap analysis, SR 11-7 alignment)

> **Data Disclaimer:** All internal company documentation references Meridian Financial Group, a fictitious entity. All data is synthetic. No real institution's policies or proprietary data are used.

📁 [View on GitHub](https://github.com/pmcavallo/PolicyLens)

---
**EvalOps: Production-Grade LLM Evaluation and Observability Platform**  
A systematic evaluation framework for LLM applications addressing non-deterministic outputs. Provides semantic similarity matching using BERT embeddings, statistical drift detection, A/B comparison with effect size calculation, and a full observability stack. Deployed on AWS via Docker with **285 tests passing** and a live demo at [http://44.213.248.8:8501](http://44.213.248.8:8501).

**Highlights:**

* **Semantic similarity:** BERT embeddings understand meaning, not just strings; "Paris" matches "The capital is Paris" correctly
* **Statistical rigor:** A/B comparison with chi-squared tests, Cohen's h effect size, and confidence intervals; not just "A looks better"
* **Drift detection:** Catch quality degradation before users complain with baseline comparison and statistical significance testing
* **Full observability:** LangSmith tracing, structured logging, metrics collection for production monitoring
* **Docker deployment:** Containerized with CPU-optimized PyTorch, published to Docker Hub, deployed to AWS EC2

**Business / Research Impact:**  
Traditional testing fails for LLMs because "correct" can be phrased a thousand ways. EvalOps provides the semantic understanding, statistical rigor, and continuous monitoring that production LLM applications require.

**Tech Stack:**

* **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
* **API / CLI:** FastAPI, Typer + Rich
* **Dashboard:** Streamlit + Plotly
* **Database:** SQLAlchemy 2.0 (SQLite/PostgreSQL)
* **Observability:** LangSmith, structlog
* **Deployment:** Docker, AWS EC2, DynamoDB

🔗 **Live Demo:** [http://44.213.248.8:8501](http://44.213.248.8:8501)  
📁 [View Full Project](https://pmcavallo.github.io/evalops/)

---

**AutoDoc AI: Multi-Agent RAG System for Regulatory Documentation**  
A production-grade multi-agent system that automates regulatory documentation generation for credit risk models. Four specialized AI agents (Research, Writer, Compliance, Editor) collaborate through custom Python orchestration to generate SR 11-7 compliant documentation with **100% source fidelity** and zero hallucinations. Presented to 200+ colleagues.

**Highlights:**

* **Four specialized agents:** Research Agent retrieves model artifacts, Writer Agent generates sections, Compliance Agent validates SR 11-7 requirements, Editor Agent ensures consistency and tone
* **Custom orchestration:** Pure Python multi-agent coordination without framework overhead, demonstrating orchestration principles from first principles
* **Source fidelity architecture:** Every claim traces to retrieved documents; system refuses to generate unsupported content
* **Production validation:** Demonstrated on real Comerica model documentation workflows; validated by credit risk stakeholders
* **Cost-benefit analysis:** Quantified time savings across documentation lifecycle with realistic adoption scenarios

**Business / Research Impact:**  
AutoDoc AI proves that regulated industries can adopt AI safely by designing for auditability first. The multi-agent architecture separates concerns (retrieval, generation, validation, editing) in ways that mirror human workflows while maintaining the traceability regulators require.

**Tech Stack:**

* **Framework / Orchestration:** Custom Python orchestration with explicit state management
* **Models / APIs:** Claude API for agent reasoning
* **Retrieval:** ChromaDB vector store with domain-specific chunking
* **Validation:** SR 11-7 compliance checklist, source citation verification
* **Presentation:** Internal demo to 200+ colleagues at Comerica

📁 [View Full Project](https://pmcavallo.github.io/AutoDoc-AI/)

---

**ChurnGuard: Production MLOps for Customer Retention**  
An end-to-end MLOps pipeline demonstrating production-grade machine learning deployment. From XGBoost model training through Docker containerization to AWS EC2 deployment, ChurnGuard showcases the full lifecycle of taking a model from notebook to production with proper experiment tracking, model registry, and API serving.

**Highlights:**

* **Full MLOps stack:** MLflow for experiment tracking and model registry, FastAPI for model serving, Docker for containerization
* **Production deployment:** Live on AWS EC2 with proper IAM, security groups, and container orchestration
* **Model performance:** XGBoost classifier with feature engineering for telecom churn prediction
* **API-first design:** RESTful endpoints for single and batch predictions with proper error handling
* **Reproducibility:** Docker ensures consistent environments from local development to cloud deployment

**Business / Research Impact:**  
ChurnGuard demonstrates that MLOps isn't about tools but about discipline: experiment tracking prevents "which model was that?", model registry enables rollback, containerization eliminates "works on my machine", and proper deployment means the model actually reaches users.

**Tech Stack:**

* **Modeling:** XGBoost, scikit-learn, pandas
* **MLOps:** MLflow (tracking + registry), FastAPI (serving)
* **Containerization:** Docker, Docker Hub
* **Cloud:** AWS EC2, IAM, Security Groups
* **Development:** Python 3.11, pytest

📁 [View Full Project](https://pmcavallo.github.io/churnguard/)

---

**MCP Banking Workflows: Model Context Protocol Tools for Regulated Industries**  
A comprehensive Model Context Protocol (MCP) server implementing nine specialized tools for banking model risk management workflows. Built to address authentic pain points in SR 11-7 compliance, cross-file consistency validation, and model dependency mapping. Demonstrates how MCP can bring AI capabilities to domain-specific enterprise workflows.

**Highlights:**

* **Nine production tools:** SAS parameter extraction, Excel dictionary parsing, cross-file consistency checking, model version comparison, Word document analysis, dependency mapping, SR 11-7 compliance checking, PowerPoint validation
* **Real workflow coverage:** Tools address actual model validation pain points: "Does the SAS code match the data dictionary?", "What changed between v3.5 and v4.2?", "Which downstream models break if we change unemployment rate inputs?"
* **950+ lines of production code:** Not a demo but a functional toolset with proper error handling and realistic outputs
* **Dependency graph analysis:** Maps upstream and downstream model relationships with effort estimates for revalidation
* **SR 11-7 compliance automation:** Checks documentation completeness against regulatory requirements

**Business / Research Impact:**  
MCP Banking Workflows shows how AI can augment (not replace) expert judgment in regulated industries. Model validators still make decisions but they spend less time on mechanical checks and more time on substantive analysis.

**Tech Stack:**

* **Protocol:** Model Context Protocol (MCP) with TypeScript SDK
* **Languages:** Python (tool implementations), TypeScript (MCP server)
* **Document Processing:** python-docx, openpyxl, python-pptx
* **Analysis:** Regex-based SAS parsing, dependency graph traversal
* **Domain:** Credit risk modeling, SR 11-7 compliance, model governance

📁 [View Full Project](https://pmcavallo.github.io/mcp-project/)

---

**CreditNLP: Fine-Tuned LLM for Startup Default Risk Prediction**  
A fine-tuned language model that identifies default risk signals in startup loan applications where traditional quantitative data is sparse. Using QLoRA on Mistral-7B, the model learns to detect implicit risk patterns in application narratives that experienced underwriters recognize intuitively but cannot codify into rules. Achieves **93.9% accuracy** on parseable outputs compared to 60% for few-shot prompting.

**Highlights:**

* **The right tool for the job:** Demonstrates when fine-tuning beats prompting and RAG; implicit pattern recognition cannot be described in prompts or retrieved from documents
* **QLoRA efficiency:** Trains only 1.1% of parameters (42M of 3.8B) on free Google Colab T4 in 41 minutes
* **Risk signal taxonomy:** Five categories (traction, financial clarity, burn rate, management, market understanding) with weighted scoring
* **Synthetic data generation:** 500 applications with controlled risk signals and known outcomes for supervised training
* **Honest evaluation:** Reports metrics on parseable outputs only (33%), distinguishing pattern recognition success from output formatting challenges

**Business / Research Impact:**  
CreditNLP proves that domain expertise can be encoded into model weights through labeled examples. The patterns experienced underwriters "feel" after thousands of applications can be learned by a 7B model in 41 minutes with the right training data.

**Tech Stack:**

* **Base Model:** Mistral-7B-Instruct-v0.3
* **Fine-Tuning:** QLoRA (4-bit quantization + LoRA adapters)
* **Libraries:** Transformers, PEFT, bitsandbytes, TRL
* **Training:** Google Colab T4 (free tier)
* **Evaluation:** Confusion matrix, precision/recall, comparison to Claude baseline

📁 [View Full Project](https://pmcavallo.github.io/creditnlp/)

---

**Fraud RT Sandbox: Real-Time Fraud Simulation & Detection**  
A sandbox environment to simulate and detect fraud in real time, combining streaming pipelines, hybrid detection logic, and dynamic responses—built to explore latency, model drift, decision rules, and system robustness under adversarial patterns.

**Highlights:**  
- **Streaming simulation:** Generates synthetic transactional data in real time to stress-test detection pipelines  
- **Hybrid detection logic:** Blends rule-based filters and ML models to flag anomalies early  
- **Drift & adaptation:** Monitors feature drift, threshold shifts, and automatically flags model retraining needs  
- **Response orchestration:** Simulates downstream actions (alerts, blocking, review queues) and tracks feedback loop  
- **Adversarial robustness experiments:** Injects attack patterns to test system resilience and false-positive trade-offs  

**Business / Research Impact:**  
This sandbox helps practitioners see how fraud detection systems behave under realistic pressure: how fast models decay, how rules must adapt, and how to balance responsiveness vs false alarms in live applications.

**Tech Stack:**  
- **Modeling & Detection:** Scikit-learn / XGBoost / rule-based logic  
- **Monitoring & Drift Detection:** Statistical tests, windowed analytics, feedback loops  
- **Simulation Tools:** Synthetic data generators, adversarial pattern injectors  
- **Orchestration & Logic:** Python pipelines, thresholding modules, alert subsystems  

📁 [View Full Project](https://pmcavallo.github.io/fraud-rt-sandbox/)

---


**Zero-Hallucination RAG Agent: Custom vs Pre-Built Tools**  
Built to answer queries about your portfolio, this project compares off-the-shelf RAG tools (like Flowise) against a custom LangChain architecture designed to **prevent hallucination at the system level**, separating metadata vs semantic paths, strict grounding, and validation.

**Highlights:**  
- **Metadata-first routing:** Project titles and metadata live outside the LLM path, eliminating hallucination in factual queries  
- **Hybrid query paths:** Distinguish metadata queries (direct lookup) vs semantic queries (LLM + retrieval)  
- **Strict grounding & validation:** System prompt enforces rules (answer only from context; cite sources; admit “I don’t know”)  
- **Automatic database resilience:** Auto-rebuild logic detects vector-store mismatches and recovers on startup  
- **0% hallucination rate in tests:** Created queries that exposed hallucination in Flowise; custom agent answered reliably with citations and refusal when out of scope  

**Business / Research Impact:**  
In contexts where credibility is essential (e.g. portfolio Q&A, knowledge systems), a system design that prevents hallucination is far more valuable than one that sounds fluent but fabricates answers.

**Tech Stack:**  
- **Framework / Libraries:** LangChain (custom pipeline), Chroma vector store  
- **Models / APIs:** GPT-4 o-mini, text-embedding-3-small  
- **Interface / Deployment:** Gradio (web UI) deployed via Hugging Face Spaces  
- **Data & Retrieval:** Chunking (3000 char chunks, 500 overlap), MMR retrieval (top-k, threshold filtering)  
- **Infrastructure Logic:** Metadata extraction (YAML frontmatter), routing logic, response validation, auto-rebuild logic  

👉 [View Full Project](https://pmcavallo.github.io/rag-agent)

---


**Prompt Engineering Lab: From Zero-Shot to Production-Ready Systems**  
A structured deep dive into prompt engineering, evolving from naive zero-shot prompts to robust, production-grade systems by layering schema enforcement, validation, and retrieval augmentation, with iterative debugging via confusion matrices.

**Highlights:**  
- **Baseline zero-shot → stabilized pipeline:** Moved from off-the-shelf prompting to a governed, structured system.  
- **Schema enforcement & validation:** Added grammar/syntax constraints to reduce hallucination and enforce output shape.  
- **Confusion-matrix driven debugging:** Used model error analysis to refine prompts and test cases over iterations.  
- **Retrieval augmentation + grounding:** Injected external context to reduce drift and improve reliability.  
- **Prompt orchestration:** Constructed multi-step prompting chains and fallback logic for edge cases.

**Business / Research Impact:**  
This lab shows how prompt engineering can mature from prototype to dependable system, critical for any AI product that needs consistency, trust, and governance.

**Tech Stack:**  
- **Models / APIs:** OpenAI GPT models (via API)  
- **Prompting tools:** Template layers, prompt chains, fallback logic  
- **Validation / Analysis:** Confusion matrices, test suites, error analysis  
- **Retrieval / Grounding:** Vector embeddings, document retrieval modules (e.g., FAISS)  
- **Orchestration & Execution:** Python pipelines, error handling, fallback prompts  

📁 [View Full Project](https://pmcavallo.github.io/prompt-engineering/)

---

**AI-in-the-Cloud Knowledge Shootout (Perplexity vs NotebookLM)**  
An experiment comparing two AI “knowledge copilots” — Perplexity and NotebookLM — across identical cloud architecture, cost, and governance prompts. The goal: see how each tool answers with varying levels of grounding, scope, and practical signal.

**Highlights:**  
- **Controlled prompting:** Each tool received the same 6 prompts (cost, architecture, governance) with identical wording.  
- **Corpus loading vs open web:** NotebookLM was fed a curated corpus (AWS/GCP docs + your Cross-Cloud work); Perplexity searched the open web in real time.  
- **Complementary strengths:** NotebookLM excelled at structured, deep synthesis tied to defined sources. Perplexity gave concise, actionable responses with broader coverage.  
- **Trade-off lens:** The difference is the point — one behaves like a policy researcher, the other like a quick-reference field guide.  

**Business / Research Impact:**  
This shootout surfaces how knowledge tools differ in utility depending on use case — whether you want precise, source-aligned advice or agile, up-to-date pointers. For teams architecting cloud + AI systems, knowing which tool to lean on (or how to combine them) is as critical as choosing the cloud services themselves.

**Tech Stack:**  
- **Knowledge Tools:** NotebookLM, Perplexity  
- **Prompting Framework:** Controlled, identical prompts in cost, architecture, and governance domains  
- **Corpus & Data:** AWS & GCP official docs, Cross-Cloud project content  
- **Evaluation Metrics:** Grounding, completeness, depth, clarity, (reproducibility pending)  

👉 [View Full Project](https://pmcavallo.github.io/ai-in-the-cloud)

---


**RiskBench AI Coding Shootout (Code Agent / AI Coding Tools Comparison)**  
A controlled “shootout” comparing three AI coding assistants — GitHub Copilot, Claude Code, and Cursor — as they each build the same end-to-end ML pipeline. Across sequential sprints, they generate synthetic data, build and tune XGBoost models, and deploy a serving API with SHAP interpretability. The project held prompts, acceptance tests, and repo structure constant, so that differences reflect tool behavior, not environment.  

**Highlights:**  
- **Fair comparison design:** Same prompts, tests, repo structure — apples-to-apples evaluation across tools  
- **Sprint methodology:** Four structured sprints (scaffolding & data, baseline modeling, serving + interpretability, final production layer)  
- **Tool divergence insights:** Copilot was fastest but more brittle; Claude struck a balance of scaffolding + correctness; Cursor faced limitations in data generation/quota  
- **Model & dataset dynamics:** Differences emerged not just in model performance but in dataset quality, feature engineering, and error handling  
- **Serving + interpretability:** Final stage delivered a stable API (FastAPI + Uvicorn) with optional SHAP explanations for predicted outputs  

**Business / Research Impact:**  
This shootout surfaces how coding agents behave *in context*, not just in toy demos. It highlights that tool choice can influence data quality, modeling decisions, and the eventual readiness of a system. For teams exploring LLM-based development, this acts as both a benchmark and a blueprint for tool-risk awareness.  

**Tech Stack:**  
- **Languages / Frameworks:** Python, FastAPI, Uvicorn, pytest  
- **Modeling / Tools:** XGBoost, SHAP for interpretability  
- **AI Assistants:** GitHub Copilot, Claude Code, Cursor  
- **Data & Testing:** Synthetic dataset generation, uniform test suite / acceptance criteria

📁 [View Full Project](https://pmcavallo.github.io/code-agent/)

---


**Cross-Cloud AutoML Shootout**  
A benchmarking exploration comparing AWS AutoML, GCP Vertex AI, and BigQuery ML on the same dataset, revealing how each cloud’s constraints, quotas, and design philosophies shape real-world ML development.

**Highlights:**  
- **Cloud comparison as design insight:** Set up “apples-to-apples” training in AWS and GCP, but discovered that hidden quotas and infrastructure limits dominate the experience.  
- **Hybrid pivot on GCP:** Vertex AI stalled under quota limits, triggering a pivot to BigQuery ML, which offered more transparent billing and fewer hidden roadblocks.  
- **Threshold tuning & business trade-offs:** Showed how the “positive class threshold” slider can turn a weak model into a usable tool, letting business teams steer between recall and precision.  
- **Infrastructure ≠ afterthought:** Even with similar algorithmic performance, the cloud path (ease, cost, quotas) often determines whether the experiment reaches production.  

**Business Impact:**  
Cloud choices *are* part of the model. The shootout demonstrates that designing AI systems isn’t only about algorithms and datasets — it’s about navigating constraints, quotas, and trade-offs that directly affect deployment and business value.  

**Tech Stack:**  
- **Cloud Platforms:** AWS SageMaker Autopilot, GCP Vertex AI, BigQuery ML  
- **Languages / Tools:** Python, SQL, Jupyter Notebooks  
- **Modeling:** AutoML classification pipelines, logistic regression, threshold tuning  
- **Data Handling:** Cloud-native storage and query layers (S3, BigQuery tables)  
- **Monitoring & Evaluation:** ROC/PR curves, precision-recall trade-offs, threshold sliders  

👉 [View Full Project](https://pmcavallo.github.io/cross-cloud)

---

**SignalGraph (PySpark + Postgres/Teradata + Prophet)**

SignalGraph is a telecom-focused anomaly detection and forecasting project that processes large-scale 4G/5G performance data (latency, jitter, PRB utilization, packet loss) through a Spark ETL pipeline and delivers real-time network insights. It demonstrates modern data workflows—from feature engineering and anomaly flagging to forecasting and graph analytics; built for scale, transparency, and decision-making in telecom environments.

**Highlights**
- **Data & Features:** Hive-partitioned Parquet with engineered features (capacity utilization, latency thresholds, PRB saturation flags, KPI interactions).
- **Modeling & Forecasting:** Time-series forecasting with Prophet to capture latency trends and test network reliability scenarios.
- **Monitoring & Anomaly Detection:** PySpark anomaly flags for performance degradation (e.g., high PRB or latency spikes), drift tracking, and cell-level stability summaries.
- **Graph & Network Analysis:** Neo4j integration with centrality metrics (degree, PageRank, betweenness) and neighbor drill-down to trace performance impacts across connected cells.
- **Policy Sandbox:** Scenario sliders to simulate SLO trade-offs (capacity, latency, reliability), threshold tuning with triage sliders, and recalibration scenarios.

📌 *Business Impact:* Helps telecom teams detect anomalies early, forecast degradation risk, and evaluate trade-offs in policy thresholds—improving service reliability and decision-making at network scale.

**Tech Stack**
- **Languages & Libraries:** Python 3.10, PySpark 3.5.1, pandas, scikit-learn, XGBoost, Prophet, matplotlib, DuckDB, SHAP, Altair, PyArrow.  
- **Frameworks:** Streamlit UI, Spark ETL.  
- **Data Stores:** Hive-partitioned Parquet, DuckDB, Postgres/Teradata schema (warehouse view).  
- **Graph & Network Analysis:** Neo4j integration, centrality metrics (degree, PageRank, betweenness), neighbor drill-in.  
- **Explainability & Monitoring:** SHAP local/global feature attribution, threshold tuning with triage slider, SLO summaries (capacity, latency, reliability).  
- **Domain:** 4G/5G KPIs (RSRP, RSRQ, SINR, PRB utilization, latency, jitter, packet loss).  

📁 [View Full Project](https://pmcavallo.github.io/signalgraph/)

---

**NetworkIQ — Incident Risk Monitor (“One Project, Three Platforms”)**

NetworkIQ is a telecom-grade incident risk system that predicts network congestion and visualizes cell-site risk across three deployment platforms (Render, GCP Cloud Run, AWS on the roadmap). It showcases how AI-first system design can be made platform-agnostic, scalable, and portable; aligning with orchestration and enterprise deployment strategies.

**Highlights**
- **Data & Features:** Ingests network telemetry (throughput, latency, packet loss, dropped session rate) via CSV into PySpark ETL, then stores in Parquet. 
- **Modeling & Prediction:** Trains multiple classifiers—including logistic regression, random forest, and XGBoost (best performer: AUC 0.86, KS 0.42)—to detect high-risk cells. 
- **Monitoring & Explainability:** Integrates SHAP for feature attribution and PSI for drift detection; includes a model card skeleton for transparency. 
- **Multi-Cloud Orchestration:** Deploys a unified Streamlit dashboard across Render and GCP Cloud Run, with AWS App Runner deployment in progress—demonstrating full “One Project, Three Clouds” orchestration.
- **Visualization & Executive Access:** Features an interactive risk map (circle size by risk magnitude, color-coded by risk level) and integrates Gemini API to generate executive summaries, recommendations, and per-cell natural-language explanations. 
- **CI/CD & Secure Ops:** Uses GitHub Actions to deploy to GCP Cloud Run and secures secrets via Google Secret Manager.

📌 *Business Impact:* NetworkIQ accelerates incident detection (reducing MTTD), supports better customer experience proxies (NPS), and lowers cost per GB—while enabling consistent, explainable AI across multiple clouds.

**Tech Stack**
- **Languages & Libraries:** Python, PySpark, XGBoost, scikit-learn, SHAP  
- **Data Pipeline & Storage:** CSV ingestion → PySpark ETL → Parquet storage  
- **Modeling:** Logistic Regression, Random Forest, XGBoost  
- **Visualization & UI:** Streamlit with interactive risk maps overlayed on cell-site visuals  
- **Cloud Platforms:** Render deployment, GCP Cloud Run (live), AWS App Runner (roadmap)  
- **CI/CD & Security:** GitHub Actions deployment workflows, Google Secret Manager  
- **Explainability & Monitoring:** SHAP for feature insights, PSI for drift, model card for transparency  
- **AI Interpretation:** Gemini API-powered executive briefings and per-cell explanations  
- **Domain Context:** Telecom congestion KPIs—throughput, latency, loss, session drop rates  

👉 [View Full Project](https://pmcavallo.github.io/network-iq/)

---

**BNPL Credit Risk Insights Dashboard (Python + Streamlit)**

A hands-on, end-to-end BNPL risk project that turns raw lending/repayment data into an interactive decision dashboard. It demonstrates modern risk workflows—from feature engineering and modeling to monitoring and “what-if” policy simulation—built for clarity, speed, and explainability.

**Highlights**
- **Data & Features:** Synthetic BNPL portfolio with engineered signals (loan-to-income, usage rate, delinquency streaks, tenure, interactions).
- **Modeling & Explainability:** Regularized logistic/CatBoost scoring with calibration, AUC/KS, and SHAP to validate driver logic.
- **Monitoring:** Drift/stability checks (e.g., PSI), score distribution tracking, and cohort comparisons across risk segments.
- **Policy Sandbox:** Threshold sliders to simulate approval/charge-off trade-offs, segment impacts, and recalibration scenarios.

📌 **Business Impact:** Helps risk teams test policies before rollout, quantify approval vs. losses, and document governance-ready decisions.

🔗 [View Full Project](https://pmcavallo.github.io/BNPL-Risk-Dashboard/)

---

**Credit Risk Model Deployment & Monitoring (AWS + PySpark + CatBoost)**

This flagship project showcases an end-to-end credit risk modeling pipeline — from scalable data processing to cloud deployment — aligned with best practices in financial services. Built using PySpark, CatBoost, SHAP, and AWS (S3, CLI), it simulates how modern risk pipelines are deployed and monitored at scale.

The full solution includes:

- **PySpark ETL pipeline** to preprocess large-scale synthetic telecom-style credit data, with engineered risk features (CLTV, utilization bins, delinquency flags)
- **Distributed logistic regression** using PySpark MLlib to validate scalable modeling workflows and evaluate performance using AUC and KS
- **AWS S3 integration** to export Parquet-formatted model-ready data for cloud-based storage and future MLOps orchestration
- **CatBoost modeling** to improve predictive power with categorical support and built-in regularization
- **SHAP explainability** to verify that key drivers (e.g., FICO, CLTV) align with domain logic and are not proxies or artifacts
- **Segment-level analysis** comparing predicted vs actual default rates by state, identifying under- and over-prediction patterns
- **Business recommendations** on scorecard calibration, behavioral feature expansion, and future automation (e.g., Airflow, SageMaker Pipelines)

💼 **Business Impact**: This project simulates a realistic production-grade credit risk pipeline — bridging data engineering, ML modeling, and cloud deployment. It highlights how interpretability and geographic segmentation can inform policy, governance, and model recalibration.

📁 [View Full Project](https://pmcavallo.github.io/aws-flagship-project/)

---

### Telecom Churn Modeling & Retention Strategy

This project demonstrates how predictive modeling and customer segmentation can be used to drive retention strategy in a telecom context. Using a publicly available customer dataset, I developed a full churn risk pipeline.

The final solution integrates:

- **Churn prediction modeling** using Logistic Regression and XGBoost with performance comparisons (AUC ≈ 0.83)
- **SHAP explainability** to identify key churn drivers (e.g., Contract Type, Risk Exposure)
- **Scorecard simulation** converting churn risk into a 300–900 scale for business-friendly deployment
- **Customer lifetime value (CLTV) integration** to quantify revenue risk across risk bands
- **Segmentation framework** (High Churn–High Value, Low Churn–Low Value, etc.) for targeted retention campaigns
- **Drift monitoring** using Population Stability Index (PSI) to track score performance over time

💡 **Business Impact**: The project enables strategic prioritization by identifying high-risk, high-value customers at risk of churn, supporting proactive retention efforts, revenue protection, and long-term profitability.

👉 [View Full Project](https://pmcavallo.github.io/Churn-Modeling-Complete)


---

### Telecom Customer Segmentation with Python

**Objective:**  
Developed a customer segmentation model using unsupervised learning on simulated postpaid telecom data to identify actionable behavioral clusters for marketing, retention, and product strategy.

**Highlights:**
- Simulated 5,000 realistic customer profiles with usage, support, contract, and churn data
- Applied full preprocessing pipeline: one-hot encoding, feature scaling, PCA for dimensionality reduction
- Performed clustering with **K-Means** (k=4) selected via **elbow** and **silhouette analysis**
- Visualized results with PCA scatter plots, boxplots, and stacked bar charts
- Profiled each segment across spend, usage, tenure, and churn risk

**Key Findings:**

<h4>📌 Key Findings</h4>

<table>
  <thead>
    <tr>
      <th>Segment</th>
      <th>Description</th>
      <th>Strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>💬 Voice-Dominant Users</td>
      <td>High voice & intl use,<br>short tenure</td>
      <td>Add voice bundles,<br>retention plans</td>
    </tr>
    <tr>
      <td>📱 High-Usage Streamers</td>
      <td>Heavy data/streaming,<br>higher churn</td>
      <td>Promote unlimited/<br>entertainment perks</td>
    </tr>
    <tr>
      <td>💸 Low-Value Starters</td>
      <td>Low usage,<br>low tenure</td>
      <td>Grow via onboarding<br>& upselling</td>
    </tr>
    <tr>
      <td>🧭 Loyal Minimalists</td>
      <td>Long tenure, low usage,<br>least churn</td>
      <td>Reward loyalty,<br>protect margin</td>
    </tr>
  </tbody>
</table>


**Tech Stack:** `Python`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`  
**Core Skills Demonstrated:** Customer analytics, unsupervised learning, PCA, strategic interpretation, stakeholder communication

👉 [View Full Project](https://pmcavallo.github.io/telecom-segmentation/)  

---
## Customer Churn Predictor

**Goal**: Predict whether a telecom customer is likely to churn using an end-to-end machine learning pipeline.

**Description**:  
This interactive app allows users to input customer features (e.g., tenure, contract type, monthly charges) and receive a real-time churn prediction. It includes data preprocessing, feature engineering, model training, cloud deployment, and live user interaction.

- 🔗 [Live App (Render)](https://churn-prediction-app-dxft.onrender.com/)
- 💻 [GitHub Repo](https://github.com/pmcavallo/churn-prediction-app)
- 📎 Technologies: `Python`, `scikit-learn`, `Streamlit`, `joblib`, `Render`

**Screenshot**:  
![Churn Prediction App Screenshot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/streamlit_ui.png?raw=true)

#### ⚙️ Tech Stack

| Purpose           | Tool                   |
|-------------------|------------------------|
| Language          | Python 3               |
| ML Library        | scikit-learn           |
| Visualization     | seaborn, matplotlib    |
| Data Handling     | pandas, NumPy          |
| Deployment        | GitHub Pages           |


## 📶 Telecom Engagement Monitoring with Fractional Logistic Regression

This project builds a full monitoring pipeline to track **postpaid customer engagement** over time using simulated telecom data. The model uses **fractional logistic regression** to predict monthly engagement as a proportion and evaluates its stability across development and monitoring datasets.

👉 **[View Full Project Notebook](https://pmcavallo.github.io/engagement-monitoring/)**

---

### 🧰 Tech Stack

| Component            | Library / Tool         |
|---------------------|------------------------|
| Modeling             | `statsmodels` (GLM - Binomial with Logit link) |
| Data Handling        | `pandas`, `numpy`      |
| Evaluation Metrics   | `sklearn.metrics`      |
| Stability Analysis   | Custom PSI logic       |
| Visualization        | `matplotlib`           |

---

### 📌 Highlights & Findings

- **Model Performance Remains Strong**:
  - RMSE and MAE remain consistent across development and monitoring samples.
  - Calibration curves closely track the 45° reference line, confirming that predicted probabilities are well-aligned with observed engagement.

- **Population Stability (PSI) Results**:
  - Most variables, including `engagement_pred`, `age`, and `network_issues`, remained stable (PSI < 0.10).
  - Moderate shifts were observed in `tenure_months` and `avg_monthly_usage`, suggesting slight distributional drift.

- **Final Monitoring Score**:
  - A weighted score combining RMSE delta, MAE delta, and PSI indicated the model is **stable**.
  - ✅ **No immediate action needed**, but moderate PSI shifts warrant ongoing monitoring in future quarters.

- **Vintage-Level Insights**:
  - Predicted and actual engagement increased from **2023Q4 to 2025Q2**, which aligns with expected behavioral trends.

---

This project demonstrates how to proactively monitor engagement models using interpretable statistics and custom stability metrics, with outputs ready for integration into model governance workflows.

### Fraud Detection with XGBoost & SHAP

A simulated end-to-end machine learning pipeline that predicts fraudulent transactions using XGBoost and interprets the model with SHAP values.

#### Objective
Detect fraudulent transactions using synthetic data with engineered features such as transaction type, amount, time, and customer behavior patterns.

#### Key Steps

- **Data Simulation**: Created a synthetic dataset mimicking real-world credit card transactions with class imbalance.
- **Preprocessing**: Handled class imbalance with SMOTE and balanced class weights.
- **Modeling**: Trained an XGBoost classifier and optimized it via grid search.
- **Evaluation**: Evaluated using confusion matrix, ROC AUC, and F1-score.
- **Explainability**: Used SHAP (SHapley Additive exPlanations) to explain model predictions and identify top drivers of fraud.

### ⚙️ Tech Stack

| Purpose           | Tool                   |
|-------------------|------------------------|
| Language          | Python                 |
| ML Library        | XGBoost, scikit-learn  |
| Explainability    | SHAP                   |
| Data Simulation   | NumPy, pandas          |
| Visualization     | matplotlib, seaborn    |
| Deployment        | Local / GitHub         |

#### 📈 Sample Output

- 🔺 Fraud detection accuracy: ~94%
- 🔍 Top features identified by SHAP:
  - `transaction_amount`
  - `time_delta_last_tx`
  - `customer_avg_tx`

📎 [View on GitHub](https://github.com/pmcavallo/fraud-detection-project) 

---

### Airline Flight Delay Prediction with Python

A full machine learning pipeline that predicts flight delays using simulated airline data enriched with real U.S. airport codes and weather features. The project explores exploratory analysis, model training, and practical recommendations for airport operations.

#### Objective
Predict whether a flight will be delayed based on features like carrier, origin, departure time, distance, and simulated weather patterns.

#### Key Steps

- **Data Simulation**: Generated a large synthetic dataset including delay labels and airport metadata.
- **EDA**: Visualized delay patterns by airport, hour of day, and weather impact.
- **Modeling**: Trained a Random Forest classifier with class balancing and hyperparameter tuning.
- **Evaluation**: Assessed performance using confusion matrix, precision-recall, and F1-score.
- **Recommendations**: Delivered operational insights and visualized them with heatmaps and scatterplots.

#### ⚙️ Tech Stack

| Purpose           | Tool                    |
|-------------------|-------------------------|
| Language          | Python 3                |
| ML Library        | scikit-learn            |
| Visualization     | matplotlib, seaborn     |
| Simulation        | NumPy, pandas            |
| Mapping (EDA)     | Plotly, geopandas        |
| Deployment        | GitHub Pages (Markdown) |

#### 📂 Read the Full Report
📎 [View Full Project](https://pmcavallo.github.io/full-airline-delay-project/)

## 🛠️ In Progress

### 🗺️ Geospatial Risk Dashboard (Tableau)
Building an interactive Tableau dashboard to visualize public health and economic risk indicators across Texas counties.

- Skills: `Tableau`, `Data Wrangling`, `Mapping`, `Interactive Filters`

> Will be added soon...

---

## What’s Next
- Migrating model workflows into modular Python scripts
- Adding CI/CD and containerization (e.g., Docker)
- Exploring model monitoring frameworks

---

For more details, view my full [portfolio homepage](https://pmcavallo.github.io) or connect via [LinkedIn](https://www.linkedin.com/in/paulocavallo/).
