---
layout: page
title: "PolicyLens: Three-Layer AI Risk Assessment Engine"
---

PolicyLens operationalizes regulatory AI governance into structured, auditable assessments. Describe an AI use case, and it produces a Risk Committee-ready report organized by the NIST AI RMF's four functions (Govern, Map, Measure, Manage), with section-level regulatory citations, internal policy alignment scoring, and a gap analysis identifying exactly where institutional governance falls short of regulatory expectations.

**Important Disclaimer:** All internal company documentation references Meridian Financial Group, a fictitious entity created for demonstration purposes. All data is synthetic. No real institution's policies or proprietary data are used.

---

## The Problem

Financial institutions deploying AI face a growing web of regulatory expectations: the NIST AI Risk Management Framework, FHFA Advisory Bulletin 2022-02, SR 11-7 model risk guidance, and emerging GenAI-specific requirements under NIST AI 600-1. Second-line risk teams must assess each AI use case against all of these frameworks, cross-reference their own internal governance policies, and document gaps with specific citations. This process is manual, inconsistent, and difficult to scale across an enterprise with dozens of AI deployments at different stages of the lifecycle.

The result is a familiar pattern: governance assessments that vary by analyst, gaps that surface late in the process rather than during development, and remediation recommendations that lack the regulatory specificity to drive action.

## The Methodology

PolicyLens implements a three-layer compliance assessment model that mirrors how a second-line risk team actually evaluates AI deployments:

```
Layer 1: External Regulation     "What does regulation REQUIRE?"
         (NIST AI 100-1, NIST AI 600-1, FHFA AB 2022-02, SR 11-7)
                    |
                    v  gap analysis
Layer 2: Internal Policy          "What does OUR POLICY cover?"
         (Institutional AI governance & MRM documentation)
                    |
                    v  use case assessment
Layer 3: Specific Use Case        "Does THIS deployment comply?"
         (User-described AI system)
```

The gap analysis between Layer 1 and Layer 2 is the critical output. It answers a question every risk manager asks: "Where does our internal policy fall short of what regulators actually expect?"

For each use case, the engine classifies risk tier based on decision impact, population affected, data sensitivity, autonomy level, and GenAI involvement. It then retrieves regulatory requirements via section-aware RAG across four ingested frameworks, retrieves internal policy coverage from a separate document collection, and generates structured findings with specific section citations. The final output includes compliance gaps sorted by severity with actionable remediation recommendations.

![PolicyLens](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/policylens.png?raw=true) 

## Demo: Customer Service GenAI Chatbot

Running the GenAI chatbot demo case (a GPT-4 powered customer service chatbot handling account inquiries and providing financial product recommendations to retail banking customers) produces an assessment classified as **High** risk tier. The rationale identifies five key risk drivers: confabulation leading to incorrect financial guidance, processing of PII and sensitive financial data, product recommendations that could result in disparate impact across protected classes, reputational risk from customer-facing errors, and regulatory compliance concerns related to consumer protection and model risk management.

The coverage scores by RMF function reveal where governance is strongest and weakest:

| RMF Function | Coverage | Gaps |
|:-------------|:--------:|:----:|
| Govern       | 38%      | 2    |
| Map          | 38%      | 3    |
| Measure      | 50%      | 3    |
| Manage       | 30%      | 4    |
| **Overall**  | **39%**  | **8**|

**Manage scores lowest at 30%** because the internal policy framework lacks GAI-specific incident response procedures, continuous monitoring for confabulation, and third-party GAI vendor management standards. These are precisely the requirements that NIST AI 600-1 introduced for generative AI systems and that most existing governance frameworks have not yet incorporated.

![Gap Analysis](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/gap_analysis.png?raw=true)

The gap analysis is the core deliverable. All 8 gaps trace to NIST AI 600-1, the GenAI-specific companion to the base NIST AI RMF. Five are Critical severity: third-party model risk mapping, customer data privacy risk mapping, GAI-specific testing methodologies, continuous monitoring for confabulation, and GAI incident response. Three are High: GenAI-specific governance provisions, third-party GAI risk management, and GAI use case documentation. The Internal Coverage column shows where the institution's policies partially address requirements versus where they are entirely absent.

The key risks identified for this use case include confabulation (GPT-4 generating plausible but incorrect responses to financial questions), data privacy and cross-customer exposure through prompt injection or context confusion, bias and disparate impact in product recommendations, and third-party vendor risk from dependence on OpenAI's infrastructure and data handling practices.

![PolicyLens2](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/policylens3.png?raw=true) 

## Demo: Credit Decisioning with Alternative Data

Running a second demo case (an XGBoost credit decisioning model using utility payment and rent history as alternative data features for consumer lending) produces a **Critical** risk tier classification. This case triggers a different regulatory profile: ECOA, FCRA, and fair lending requirements dominate, with alternative data features flagged as potential proxies for protected classes. The assessment finds 7 critical gaps, with Govern scoring 33% due to absent fair lending governance structures for AI/ML, and an overall coverage of 49%.

The contrast between the two demos illustrates how the engine adapts its assessment to the specific risk profile of each use case. A GenAI chatbot triggers NIST AI 600-1's confabulation and content provenance requirements. A credit decisioning model triggers fair lending testing and alternative data governance requirements. The same three-layer methodology produces different findings based on which regulatory requirements are applicable.

## Frameworks Covered

The assessment engine ingests and indexes four regulatory frameworks:

**NIST AI 100-1 (AI RMF v1.0)** serves as the primary structural backbone, providing the Govern, Map, Measure, and Manage functions that organize every assessment. **NIST AI 600-1 (GenAI Profile)** adds GenAI-specific risk overlays for hallucination, data leakage, content provenance, and other risks unique to generative AI systems. **FHFA Advisory Bulletin 2022-02** provides financial services-specific AI/ML risk management requirements from the Federal Housing Finance Agency. **SR 11-7** provides the foundational model risk management guidance from the Federal Reserve and OCC that underpins all model governance in banking.

The internal policy collection includes two synthetic governance documents for Meridian Financial Group: an AI Governance Policy and a Model Risk Management Policy. These represent the institutional governance artifacts that a real second-line team would maintain.

## Architecture

The system uses section-aware RAG across dual ChromaDB collections. Regulatory documents are parsed preserving their hierarchical structure (not naive fixed-window chunking), so every retrieval carries its full section lineage for precise citations. The regulatory collection contains 251 chunks from four frameworks; the internal policy collection contains 45 chunks from two documents.

```
User Input (AI use case description)
        |
        v
Risk Tier Classifier (Claude API)
        |
        v
Dual-Collection RAG Retrieval (ChromaDB)
  - regulatory_frameworks (251 chunks)
  - internal_policies (45 chunks)
        |
        v
Assessment Generator (Claude API + structured prompt)
        |
        v
Pydantic Validation + Post-Processing
  (coverage scoring, gap sorting, citation collection)
        |
        v
Output: Streamlit UI / Executive Report / JSON
```

Assessment output is validated through Pydantic schemas enforcing structural completeness: four RMF functions present, all findings carrying citations, severity enums enforced, and coverage scores calculated from the findings data. The Streamlit UI provides an interactive dashboard with eight tabs covering the executive summary, each RMF function, gap analysis, SR 11-7 alignment, and a citations index.

![Risk Tier](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/policylens2.png?raw=true)

## Tech Stack

Python 3.10, ChromaDB (dual collections, cosine similarity), sentence-transformers (all-MiniLM-L6-v2) for embeddings, Claude API for classification and assessment generation, PyMuPDF for PDF parsing, Pydantic for output validation, and Streamlit for the interactive dashboard.

## Why This Matters for AI Risk Management

PolicyLens is not a chatbot that answers questions about regulations. It is a structured assessment engine that produces the governance artifacts a second-line risk team needs: risk tier classification with documented rationale, regulatory requirement mapping organized by RMF function, internal policy adequacy scoring with specific citations, compliance gaps ranked by severity, and remediation recommendations that reference the specific regulatory expectations being missed.

The gap analysis between external requirements and internal policy coverage is the feature that distinguishes this from a regulatory search tool. It answers the question that matters most during AI deployment reviews: not "what does the regulation say?" but "where does our governance framework fall short of what the regulation requires?"

---

**Repository:** [github.com/pmcavallo/PolicyLens](https://github.com/pmcavallo/PolicyLens)

**Built with:** Python, ChromaDB, Claude API, Streamlit

**Author:** Paulo Cavallo, PhD
