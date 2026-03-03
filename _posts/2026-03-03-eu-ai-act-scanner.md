---
layout: page
title: "EU AI Act Compliance Scanner: Three AI Agents Building a Regulatory Tool"
---

The EU AI Act Compliance Scanner takes a plain-English description of how an organization uses AI and produces a full compliance assessment: risk classification, deployer-to-provider role analysis, GPAI provider terms check, obligation gap analysis, and prioritized next steps. It runs entirely offline with no LLM dependency, and the entire project was built by three AI agents working in parallel through Claude Code Agent Teams.

This page covers the regulatory problem, how the scanner works, how Agent Teams built it, and what the experience revealed about multi-agent AI development.

---

## The Problem

The EU AI Act introduces a risk-based framework for AI systems that takes effect in stages through 2026-2027. The high-risk provisions — the ones with real teeth — apply starting August 2, 2026. Organizations using AI in employment, credit scoring, insurance, education, law enforcement, or critical infrastructure face significant compliance obligations.

The dangerous gap is Article 25. It defines three scenarios where an organization that thinks it is merely a "deployer" (light obligations) is actually a "provider" (full compliance regime). The most common trap is Scenario C: using a general-purpose AI system like ChatGPT, Claude, or Copilot for a purpose that falls under Annex III high-risk categories. An HR team that uses GPT-4 to screen resumes has just made their organization a provider of a high-risk AI system. Most companies have no mechanism to detect this.

No self-service tool existed to check: *"Given how we use AI, what is our classification, what is our role, and what do we owe?"*

This scanner fills that gap.

---

## What the Scanner Does

The scanner accepts a natural language description of an AI use case and runs it through a five-stage classification pipeline:

**Stage 1 — GPAI Detection.** Regex patterns identify whether the description references a general-purpose AI product (GPT-4, Claude, Gemini, Copilot, Llama, etc.) and which provider it belongs to.

**Stage 2 — Annex III Matching.** A keyword scoring engine matches the description against all 8 high-risk categories and their subcategories. Each category has a curated keyword list, domain-specific boosting patterns, and example-based overlap scoring. The system correctly handles the Annex III Category 5a fraud detection exclusion — AI systems used for detecting financial fraud are explicitly carved out from the credit scoring high-risk classification.

**Stage 3 — Risk Level Determination.** Based on the Annex III match result, the system assigns one of four risk levels: Unacceptable (banned under Article 5), High-Risk (full compliance regime), Limited Risk (transparency obligations only), or Minimal Risk (no obligations).

**Stage 4 — Article 25 Analysis.** If a GPAI product was detected and an Annex III match was found, the scanner checks all three deployer-to-provider role shift scenarios:

| Scenario | Trigger | Detection Method |
|----------|---------|------------------|
| Article 25(1)(a) — Rebranding | Organization puts their name/trademark on a high-risk AI system | Keywords: "white-label," "rebrand," "our brand," "offer under" |
| Article 25(1)(b) — Substantial Modification | Organization significantly changes the AI system | Keywords: "fine-tune," "retrain," "modify," "our own data" |
| Article 25(1)(c) — Modified Intended Purpose | Organization uses GPAI for a high-risk purpose | Combination: GPAI detected + Annex III match |

Scenario C is the most dangerous because organizations trigger it unknowingly. The scanner flags it at 95% confidence when both conditions are met.

**Stage 5 — GPAI Terms Check.** If a GPAI provider was detected and the use case is high-risk, the scanner cross-references against curated summaries of each provider's acceptable use policy. Using GPT-4 for resume screening likely violates OpenAI's terms. Using Claude for credit scoring likely violates Anthropic's terms. The scanner identifies the specific terms at risk.

**Output.** The pipeline produces a structured compliance report with:

- Risk classification summary with confidence scores
- Article 25 analysis with plain-English reasoning
- GPAI provider terms violation details
- Compliance gap table mapping all applicable obligations to action items
- Executive summary for leadership
- Prioritized next steps (Critical → High → Medium → Low)
- Professional HTML report with color-coded risk banners

---

## Architecture

```
                    Plain-English Description
                              │
                              ▼
                    ┌───────────────────┐
                    │  GPAI Detection   │  Regex pattern matching
                    │  (Stage 1)        │  against 5 providers
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │  Annex III Match  │  Keyword scoring across
                    │  (Stage 2)        │  8 categories + exclusions
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │  Risk Level       │  Unacceptable → High →
                    │  (Stage 3)        │  Limited → Minimal
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │  Article 25       │  3 role-shift scenarios
                    │  (Stage 4)        │  with confidence scoring
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │  GPAI Terms Check │  Provider AUP cross-ref
                    │  (Stage 5)        │  for high-risk uses
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │  Report Generator │  Obligation gaps +
                    │  + Priority Scorer│  executive summary +
                    │  + HTML Template  │  prioritized next steps
                    └───────────────────┘
                             │
                             ▼
                    Compliance Report (HTML)
```

The entire pipeline is rule-based. No LLM calls at runtime. No API keys required. Same input always produces the same output. This is a deliberate design choice: for a compliance tool, deterministic and reproducible output is more valuable than LLM-generated analysis.

---

## Demo: HR Resume Screening with GPT-4

To show what the scanner produces end-to-end, here is a complete scan of a realistic scenario: an HR team using GPT-4 via API to screen resumes and rank candidates.

### Input and Classification

![Gradio](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/screenshot-gradio-ui.png?raw=true) 

The scanner accepts the plain-English description on the left and produces the classification summary on the right. In this case: HIGH RISK, PROVIDER role, Annex III Category 4a (Employment), Article 25(1)(c) triggered, OpenAI terms violated, risk score 100/100, and 0 of 10 obligations met.

### Compliance Report

![Report](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/screenshot-report-summary.png?raw=true) 

The HTML report opens with a color-coded risk banner and a grid summary. The 45% classification confidence reflects the keyword overlap score between the input description and Category 4a's keyword list. The risk score of 100/100 is driven by the combination of high-risk classification, provider role via Article 25, and zero obligations met.

### Article 25 Analysis

![Report](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/screenshot-article-25.png?raw=true) 

This is the section most organizations miss. The scanner detected that a GPAI product (OpenAI's GPT-4) is being used for an Annex III high-risk purpose (employment decisions), which triggers Article 25(1)(c). The consequence is spelled out in plain English: your organization now bears full provider obligations under Articles 8-15 and Article 47. The original provider (OpenAI) must cooperate and provide necessary information under Article 25(2).

### Compliance Gap Analysis

The full report includes a compliance gap table mapping all 10 provider obligations to their regulatory articles, priority level, and specific action items. For the HR resume screening scenario, all 10 obligations are flagged as not met:

| Obligation | Article | Priority | Status |
|-----------|---------|----------|--------|
| Risk Management System | Article 9 | CRITICAL | Not Met |
| Data and Data Governance | Article 10 | CRITICAL | Not Met |
| Technical Documentation | Article 11 | CRITICAL | Not Met |
| Record-Keeping / Automatic Logging | Article 12 | HIGH | Not Met |
| Transparency and Provision of Information | Article 13 | HIGH | Not Met |
| Human Oversight | Article 14 | CRITICAL | Not Met |
| Accuracy, Robustness and Cybersecurity | Article 15 | HIGH | Not Met |
| Conformity Assessment | Article 43 | CRITICAL | Not Met |
| EU Declaration of Conformity | Article 47 | HIGH | Not Met |
| Registration in EU Database | Article 49 | HIGH | Not Met |

### Prioritized Next Steps

The report closes with a prioritized action list. For the HR resume screening scenario:

1. **[CRITICAL]** Address Article 25 role shift: your organization has become a provider due to modified intended purpose. Full provider obligations now apply.
2. **[CRITICAL]** Review and address OpenAI terms of use violations. Current usage may breach the provider's acceptable use policy.
3. **[CRITICAL]** Risk Management System (Article 9): Establish a documented risk management system covering the AI system's full lifecycle.
4. **[CRITICAL]** Data and Data Governance (Article 10): Create a data governance policy covering collection, preparation, labelling, and bias examination.
5. **[CRITICAL]** Technical Documentation (Article 11): Draft technical documentation per Annex IV before the system is placed on the market.
6. **[CRITICAL]** Human Oversight (Article 14): Implement a human-in-the-loop mechanism so operators can review, override, or halt AI outputs.
7. **[CRITICAL]** Conformity Assessment (Article 43): Determine whether internal or third-party conformity assessment applies (third-party for biometrics).
8. **[HIGH]** Record-Keeping / Automatic Logging (Article 12): Design automatic logging capabilities into the system architecture.
9. **[HIGH]** Transparency and Provision of Information to Deployers (Article 13): Prepare instructions for use that describe system capabilities, limitations, and intended purpose.
10. **[HIGH]** Accuracy, Robustness and Cybersecurity (Article 15): Define and publish accuracy benchmarks for the intended use case.
11. **[HIGH]** EU Declaration of Conformity (Article 47): Draft an EU declaration of conformity stating compliance with Chapter III Section 2 requirements.
12. **[HIGH]** Registration in EU Database (Article 49): Register the high-risk AI system in the EU database (Article 71) before market placement.

Every item maps to a specific EU AI Act article. The scanner doesn't give you vague advice — it tells you exactly which obligation you're missing and what the regulation requires.

---

## Test Results

8 test scenarios covering the classification edge cases that matter:

| Scenario | Expected | Result |
|----------|----------|--------|
| HR resume screening with GPT-4 | High-risk, Provider (Art 25(1)(c)), OpenAI terms violated | ✅ Pass |
| Bank uses vendor credit scoring as-is | High-risk, Deployer, no Article 25 trigger | ✅ Pass |
| Company rebrands AI hiring tool | High-risk, Provider (Art 25(1)(a)) | ✅ Pass |
| Fine-tuned open-source fraud model | NOT high-risk (fraud exclusion), Provider (Art 25(1)(b)) | ✅ Pass |
| Marketing copywriting with Claude | Limited risk, Deployer, no terms violation | ✅ Pass |
| University admissions chatbot with Gemini | High-risk, Provider (Art 25(1)(c)), Google terms violated | ✅ Pass |
| Hospital triage with GPT-4 | High-risk, Provider (Art 25(1)(c)), OpenAI terms violated | ✅ Pass |
| Customer service chatbot, no decisions | Limited risk, Deployer | ✅ Pass |

Test scenario 4 (fraud detection) is the critical edge case. Annex III Category 5a explicitly excludes AI systems used for detecting financial fraud from the credit scoring high-risk classification. The classifier correctly identifies this exclusion and classifies the system as not high-risk, even though it detects the Article 25(1)(b) substantial modification trigger. This matters because a naive keyword matcher would flag "transaction classification" as credit-related and incorrectly classify it as high-risk.

Full test suite: **109 collected, 103 passed, 6 skipped, 0 failed.** The 6 skipped tests are non-high-risk scenarios correctly bypassed by obligation gap tests (minimal and limited risk systems have no provider/deployer obligations to test).

Custom input validation: Beyond the pre-built scenarios, the scanner correctly classifies novel descriptions not in the test set. An insurance premium pricing use case with Claude correctly matched Category 5b (insurance), detected Anthropic as GPAI provider, flagged terms violation, and triggered Article 25(1)(c) — all from a description written after the system was built.

---

## How Agent Teams Built This

This project was built using Claude Code's experimental Agent Teams feature, shipped by Anthropic on February 5, 2026 alongside the Opus 4.6 release. Agent Teams allows multiple Claude Code instances to work in parallel as a coordinated team, with a shared task list, direct inter-agent messaging, and a lead agent that orchestrates the work.

### The Team

| Agent | Role | Files Owned | Output |
|-------|------|-------------|--------|
| **Lead** (Claude Code) | Orchestrator — set up tasks, dependencies, verified outputs, built UI and tests | `app.py`, `src/shared/`, `tests/` | Gradio UI (128 LOC), 3 test files (520 LOC) |
| **Regulatory Researcher** | Verified and enriched all regulatory JSON data | `data/regulations/`, `data/provider_terms/` | 4 enriched JSON files |
| **Classifier Agent** | Built the rule-based classification engine | `src/classifier/` | `rules.py` (565 LOC), `classifier.py` (181 LOC) |
| **Report Writer** | Built report generation pipeline and HTML template | `src/reports/` | Generator, priority scorer, Jinja2 template |

### The Build Process

**Step 1: Scaffolding (before Agent Teams).** The project structure, CLAUDE.md with agent role definitions and file boundaries, regulatory data files, shared type definitions, and test scenarios were prepared in advance. This gave the agents a running start — they didn't need to research the EU AI Act from scratch. The CLAUDE.md file is the critical coordination artifact. It defines who owns which directories, what the execution order should be, and what the key regulatory concepts are. Agent Teams teammates load CLAUDE.md automatically when they spawn but do NOT inherit the lead's conversation history. Everything they need must be in files.

**Step 2: Task dependency chain.** The Lead created four tasks with explicit dependencies:

```
Task 1: Regulatory Data (unblocked — starts immediately)
    ↓
Task 2: Classifier Engine (blocked on Task 1)
Task 3: Report Generator (blocked on Task 1)
    ↓
Task 4: Gradio UI + Tests (blocked on Tasks 2 and 3)
```

This is real orchestration, not just "run three things at once." The Regulatory Researcher had to deliver first because the other agents depended on verified data files.

**Step 3: Parallel execution.** Once the Regulatory Researcher completed Task 1, the Classifier Agent and Report Writer unblocked automatically and worked in parallel. Each agent operated in its own context window with its own directory. No merge conflicts, no file collisions.

**Step 4: Verification before integration.** Before writing a single line of UI code, the Lead ran all 8 test scenarios through the classifier manually. Every scenario passed. Only then did the Lead proceed to build `app.py` and the test suite.

**Step 5: Full test run.** 109 tests collected, 103 passed, 6 skipped, 0 failed. The Lead cleaned up the team and delivered the final summary.

### Timeline

Total build time from team creation to all tests passing: **approximately 15 minutes.**

### What This Reveals About Agent Teams

**File boundaries are essential.** The CLAUDE.md file defined strict directory ownership. Without this, agents would have edited each other's files and caused conflicts. This is the same principle as microservice boundaries — clear ownership prevents coordination overhead.

**Preparation matters more than prompting.** The agents were productive because the regulatory data, shared types, and test scenarios were already in place. They built on a foundation rather than starting from zero. The better your scaffolding, the better the agent output.

**The lead must wait.** The lead's instinct is to start building immediately. Forcing it to wait for teammates to deliver, then verify, then build — that's the discipline that makes agent teams work. Without it, you get a lead that implements everything itself and the teammates become expensive observers.

**Deterministic output is a feature.** The agents chose to go fully rule-based rather than adding an LLM classification layer. For a compliance tool, this is correct — same input always gives the same output. The rule-based approach has coverage limitations on vague inputs, but for concrete AI system descriptions, it works reliably.

**Cost structure.** This build ran on a Claude Max subscription (flat rate). No per-token API costs for the Agent Teams build itself. The scanner runs with zero token cost per scan since it makes no LLM calls at runtime.

---

## Regulatory Data Coverage

The scanner's knowledge base covers:

**Annex III — 8 High-Risk Categories:**
1. Biometrics (remote identification, emotion recognition)
2. Critical infrastructure (energy, water, transport, digital)
3. Education and vocational training (access, assessment, monitoring)
4. Employment (recruitment, screening, evaluation, termination)
5. Essential services (credit scoring, insurance, emergency dispatch, public benefits)
6. Law enforcement (risk assessment, lie detection, profiling, predictive policing)
7. Migration and border control (risk assessment, document verification)
8. Administration of justice (legal research, sentencing, election influence)

**Article 25 — 3 Role-Shift Scenarios:**
- Scenario A: Rebranding a high-risk AI system
- Scenario B: Substantially modifying an AI system
- Scenario C: Using GPAI for a high-risk purpose (the trap most organizations fall into)

**Provider Obligations (Articles 8-15, 43, 47, 49):** Risk management system, data governance, technical documentation, automatic logging, transparency, human oversight, accuracy/robustness/cybersecurity, conformity assessment, EU declaration, database registration.

**Deployer Obligations (Article 26):** Use per instructions, human oversight, input data relevance, monitoring, record keeping, worker notification, DPIA.

**Transparency Obligations (Article 50):** AI interaction disclosure, synthetic content marking, emotion recognition/biometric categorisation notification.

**GPAI Provider Terms:** OpenAI, Anthropic, Google, Microsoft, Meta — curated summaries of acceptable use restrictions relevant to high-risk deployments.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Web UI | Gradio |
| Report templates | Jinja2 |
| Testing | pytest |
| Classification | Rule-based (regex + keyword scoring) |
| Build tool | Claude Code Agent Teams |
| LLM dependency at runtime | None |

---

## Limitations

- **Keyword coverage.** Very vague descriptions ("we use AI for some HR tasks") may not trigger a high-risk match. Concrete descriptions work reliably.
- **No LLM fallback.** Ambiguous edge cases that would benefit from LLM reasoning are classified by keyword scoring alone. A future version could add an optional LLM layer for borderline cases.
- **Provider terms are summaries.** The GPAI provider terms data is curated from public acceptable use policies, not a legal analysis. Terms change. Check original documents for current versions.
- **All obligations default to "not met."** The scanner assumes no existing compliance measures are in place. A future version could accept user input on current compliance status.
- **EU Commission guidelines pending.** The definition of "substantial modification" (Article 25(1)(b)) may be clarified by forthcoming EU Commission guidance. The scanner's detection criteria may need updating.
- **Educational purpose.** This is a portfolio project demonstrating regulatory knowledge and AI engineering. It is not legal advice.

---

## Connection to Other Work

This project extends a body of work at the intersection of AI engineering and regulatory compliance in financial services:

| Project | Relationship |
|---------|-------------|
| [MCP Banking Workflows](https://pmcavallo.github.io/mcp-project/) | MCP server for model risk management automation. Same domain (regulated AI), different protocol (MCP vs. standalone). |
| [EvalOps](https://pmcavallo.github.io/evalops/) | LLM evaluation framework. Tests model correctness; the scanner tests regulatory compliance. |
| [CreditNLP](https://pmcavallo.github.io/creditnlp/) | Fine-tuned Mistral-7B for credit risk. Would be classified as high-risk under Annex III 5a by this scanner. |
| [AI Under Audit Newsletter](https://pmcavallo.github.io) | Issue 008 ("The EU AI Act Already Applies to You") established the regulatory thesis. This tool makes it actionable. |

---

## Links

- **GitHub Repository:** [github.com/pmcavallo/eu-ai-act-scanner](https://github.com/pmcavallo/eu-ai-act-scanner)
- **Newsletter (Issue 008):** The regulatory context behind this project
- **Claude Code Agent Teams Documentation:** [code.claude.com/docs/en/agent-teams](https://code.claude.com/docs/en/agent-teams)

---

*All scenarios use synthetic/simulated data. No real company names or proprietary information. This project is a portfolio demonstration of regulatory knowledge and AI engineering capabilities, not legal advice.*

*Built by Paulo Cavallo, PhD — Senior AI Orchestrator | AI Governance in Regulated Industries*
