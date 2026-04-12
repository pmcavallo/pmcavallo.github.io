---
layout: default
title: "AuditHive: AI Governance Middleware"
---

# AuditHive: AI Governance Middleware

Mid-sized companies are deploying AI faster than they can govern it. A chatbot tells a customer the wrong refund policy. An AI-drafted email recommends an unsuitable investment. A document generator leaks confidential data. In each case, the company finds out after the damage is done, from Twitter, from a regulator, or from a lawsuit.

Enterprise governance platforms exist (Credo AI, IBM watsonx.governance, Arthur AI), but they require six-figure contracts, months of implementation, and dedicated compliance teams. Open-source toolkits exist (Microsoft Agent Governance Toolkit, Guardrails AI), but they require engineers to deploy and configure. Policy template PDFs exist, but they sit in binders while the AI runs unsupervised.

Nobody serves the company with 50 to 500 employees that deployed AI last quarter and hasn't thought about what happens when it makes a mistake.

AuditHive fills that gap. It's a middleware layer that sits between your application and your LLM provider. Change one API endpoint. Every LLM call is intercepted, policies are enforced, and everything is logged. But enforcement is the table stakes, the part any engineering team could build. What makes AuditHive different is what it tells you that you didn't know to ask.

> **[Live Demo →](https://pmcavallo.github.io/AuditHive/)**
> 
> The demo uses pre-built example data simulating an e-commerce company with a customer-facing chatbot. No backend required.

---

## The Dashboard

### Overview

The overview gives a VP of Operations or a Head of Compliance a single screen that answers: "What is my AI doing, and should I be worried?"

Four metric cards show total calls, violations blocked, interactions flagged, and active policies. A line chart tracks call volume and violation trends over the past seven days. Below it, a table surfaces the most recent violations sorted by severity, each one linked to the full audit trail.

The governance coverage score and maturity level badge connect enforcement data to the assessment engine: not just "what happened this week" but "how governed are we overall."

![AuditHive Overview Dashboard](https://raw.githubusercontent.com/pmcavallo/AuditHive/main/Images/overview.png)

### Governance Assessment

This is the feature that changes the conversation from "here's your audit trail" to "here's what you're missing."

The customer answers four onboarding questions: what their AI does, who uses it, what industry they're in, and where their users are located. From those answers, AuditHive maps applicable regulations, identifies required controls, computes a coverage score, and generates the specific questions a regulatory examiner would ask about the gaps.

The assessment page shows applicable regulations with reasons ("Your customers include Colorado residents"), each required control with covered or missing status, a governance maturity score based on the Four Questions framework (Purpose, Working, Failure, Accountability), and described-vs-established gaps where policy configuration doesn't match enforcement reality.

![Governance Assessment](https://raw.githubusercontent.com/pmcavallo/AuditHive/main/Images/gov_assess.png)

### Audit Trail

Every LLM interaction flows through an immutable audit trail. The audit trail page provides a searchable, filterable, paginated view of every call: timestamp, status (completed, blocked, flagged), model used, a preview of the user message, policy violations detected, and response latency.

Status badges color-code instantly: green for completed, red for blocked, yellow for flagged. Clicking a row expands to show the full request, the full response, every policy check that ran with its result, and the complete metadata.

The audit trail is the artifact a compliance officer needs for examination readiness. Not a summary. Not a dashboard metric. The complete record.

![Audit Trail](https://raw.githubusercontent.com/pmcavallo/AuditHive/main/Images/audit_trail.png)

### Regulatory Timeline

AI regulation is moving fast. Colorado's AI Act enforcement begins June 2026. The EU AI Act's high-risk rules take effect August 2026. California's AI Transparency Act becomes enforceable the same month. New York's RAISE Act is in the pipeline.

The regulatory timeline page doesn't just list these deadlines. It tells each customer whether a regulation affects their specific setup, based on their industry, jurisdictions, AI use cases, and current policy configuration. A Colorado AI Act deadline shows as "critical" for a customer with Colorado users and no AI disclosure configured, but "low" for a customer who already has disclosure enabled.

Each timeline entry includes: what changed, why it matters for this customer, what controls are missing, the deadline, and the fix.

![Regulatory Timeline](https://raw.githubusercontent.com/pmcavallo/AuditHive/main/Images/reg_timeline.png)

---

## What Sets It Apart

AuditHive was designed around a question that doesn't appear in any product in this category: **"What would a regulatory examiner ask about your AI governance, and can you answer?"**

Most governance tools provide enforcement. Block PII. Detect injection. Log everything. These are necessary but not sufficient. An examiner doesn't ask "do you have a proxy?" They ask "show me your governance framework, show me how you know it's working, and show me what happens when it breaks."

Five capabilities in AuditHive were built specifically to answer those questions. They come from the experience of sitting across the table from model validators and regulatory examiners, not from reading a product requirements document:

**Governance Assessment Engine.** Four onboarding questions produce a complete regulatory gap analysis. The engine maps combinations of industry, jurisdiction, AI use case, and audience type to a database of applicable regulations and required controls. The output isn't "here are some best practices." It's "you are subject to these 7 regulations, you need these 12 controls, you have 4 configured, and here are the 8 gaps."

**Examiner Simulation.** A weekly digest that doesn't say "12 violations this week." It says "if an examiner reviewed this week, here's what they'd ask, and here's the answer you'd give based on your audit trail." Powered by an LLM agent with a system prompt that encodes how examiners actually think, prioritize, and frame findings. With a deterministic fallback for environments without LLM access.

**Described-vs-Established Gap Detection.** Most companies have described governance: a policy document that says what they do. Few have established governance: controls that actually enforce it. AuditHive continuously compares what the policy configuration says against what the enforcement data shows. "Your policy says block PII. But email addresses are set to flag, not block. 47 messages passed this month. That's a governance gap an examiner would catch."

**Four Questions Framework.** Every governed AI system should answer: What is its purpose? How do you know it's working? What happens when it breaks? Who is accountable? AuditHive asks these questions during onboarding, cross-references the answers against actual system configuration, and produces a governance maturity score. If the customer says "we have alerts" but no alerting is configured, the framework catches the mismatch.

**Proactive Regulatory Impact Assessment.** Not a news feed. A personalized assessment that says "Colorado AI Act enforcement begins June 30. YOUR chatbot doesn't have AI disclosure enabled. YOUR customers include Colorado residents. Here's the one-click fix and the deadline."

---

## Architecture

```
Customer Application
    │ (one API endpoint change)
    ▼
┌────────────────────────────────────────────┐
│            AUDITHIVE MIDDLEWARE             │
│                                            │
│  ┌──────────────┐    ┌──────────────────┐  │
│  │ Pre-call     │◄───│ Policy Templates │  │
│  │ Policy Engine│    │ (regulatory-     │  │
│  │ (PII, inject,│    │  grounded)       │  │
│  │  content,    │    └──────────────────┘  │
│  │  scope)      │                          │
│  └──────┬───────┘    ┌──────────────────┐  │
│         │            │ Assessment Engine │  │
│         ▼            │ (regulatory map, │  │
│  ┌──────────────┐    │  gap detector,   │  │
│  │ LLM Provider │    │  four questions, │  │
│  │ (passthrough)│    │  maturity score) │  │
│  └──────┬───────┘    └──────────────────┘  │
│         │                                  │
│         ▼            ┌──────────────────┐  │
│  ┌──────────────┐    │ Examiner Agent   │  │
│  │ Audit Logger │    │ (LLM-powered     │  │
│  │ (encrypted,  │    │  with static     │  │
│  │  immutable)  │    │  fallback)       │  │
│  └──────┬───────┘    └──────────────────┘  │
│         │                                  │
│         ▼            ┌──────────────────┐  │
│  ┌──────────────┐    │ Regulatory Intel │  │
│  │ Alert        │    │ (impact assess,  │  │
│  │ Dispatcher   │    │  timeline,       │  │
│  │ (email,      │    │  auto-alert)     │  │
│  │  webhook)    │    └──────────────────┘  │
│  └──────────────┘                          │
│         │                                  │
└─────────┼──────────────────────────────────┘
          ▼
   Dashboard + Reports + Governed Response
```

**Integration is one line of code:**

```python
# Before (direct to OpenAI):
client = OpenAI(api_key="sk-...")

# After (through AuditHive):
client = OpenAI(
    api_key="ah-...",
    base_url="http://localhost:8000/v1",
    default_headers={"X-LLM-Key": "sk-..."}
)
```

The customer's LLM API key passes through in memory only. Never stored. Never logged. The customer's data stays on their infrastructure. AuditHive is designed for self-hosted deployment: the customer runs it on their own servers via Docker Compose. No data leaves their environment.

---

## Security

AuditHive handles sensitive data (every prompt and response in the audit trail), so security was built as a first-class concern, not an afterthought:

**Encryption at rest.** Audit log content (prompts and responses) is encrypted with AES-256-GCM before writing to the database. Each installation generates its own encryption key. Even with database access, conversation content is unreadable without the key.

**Tenant isolation.** PostgreSQL Row-Level Security prevents cross-tenant data access at the database level, in addition to application-level filtering. Defense in depth: a bug in application code cannot leak data across tenants.

**API key safety.** The customer's LLM provider API key exists in memory only during the proxy request. It is never written to the database, never appears in logs, never appears in error messages. Dedicated security tests verify this.

**AI features opt-in.** The examiner simulation can use an LLM (Claude) for richer analysis. This feature defaults to OFF. The customer must explicitly enable it with a disclosure explaining what data is sent. A deterministic static fallback works without any external API calls.

**Data lifecycle.** Full data export (all customer data as JSON) and permanent deletion endpoints. GDPR Article 17 (right to erasure) and Article 20 (right to portability) compliant by design.

---

## Policy Templates

AuditHive ships with pre-built policy templates grounded in actual regulation, not generic best practices:

| Template | Risk Level | Regulatory Grounding |
| --- | --- | --- |
| Customer-facing chatbot | High | FTC chatbot guidance, Colorado AI Act, California SB 243, EU AI Act Article 52, NIST AI RMF |
| Document generation | Medium | CCPA/GDPR, NIST AI 600-1, ISO 42001, SR 11-7 |
| Email automation | Medium | CAN-SPAM Act, FTC deceptive practices, California AI Transparency Act |

Each template defines pre-call checks (PII detection, prompt injection blocking, content filtering, scope enforcement) and post-call evaluation, with configurable actions (block, flag, log) and thresholds. Customers apply a template and have governance active immediately, then customize as needed.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend API | FastAPI, Python 3.11+, async |
| Data models | Pydantic v2 (strict validation), SQLAlchemy 2.0 |
| Database | PostgreSQL 16+ with Row-Level Security |
| Encryption | AES-256-GCM (cryptography library) |
| Dashboard | React 18, Tailwind CSS, Recharts, Vite |
| Examiner Agent | Anthropic Claude (with deterministic fallback) |
| Alerting | SMTP email + HTTP webhook |
| Reports | Jinja2 HTML templates (print-optimized) |
| Testing | pytest, 162 tests passing |

---

## Build History

AuditHive was built in 9 phases with comprehensive test coverage and zero regressions across all phases:

| Phase | What | Tests |
| --- | --- | --- |
| 1 | Foundation: FastAPI scaffold, PostgreSQL, auth, API keys | 11 |
| 2 | Core middleware: OpenAI-compatible proxy, policy engine, audit logging | 44 |
| 3 | Policy templates: 3 templates, deep-merge customization, browse/apply API | 22 |
| 4 | Dashboard: React + Tailwind, 8 pages, Recharts charts | 5 |
| 5 | Assessment engine, gap detector, four questions framework, 15 regulatory mappings | 24 |
| 6 | Examiner simulation (LLM agent + static fallback), HTML reports, email/webhook alerting | 20 |
| 7 | Proactive regulatory impact assessment, timeline, auto-alert on new regulations | 18 |
| 8 | Security: AES-256-GCM encryption, Row-Level Security, GDPR data export/delete, AI opt-in | 18 |
| 9 | Demo mode with pre-built data, GitHub Pages deployment | 0 (frontend) |
| **Total** | **30+ API endpoints, 10 dashboard pages, 15 regulatory mappings, 7 regulatory updates** | **162** |

Detailed phase completion summaries with design decisions, test matrices, and architecture notes are available in the [repository](https://github.com/pmcavallo/AuditHive) under `docs/phases/`.

---

## Synthetic Data Disclaimer

All data in this project is entirely **synthetic**. Company names, customer personas, audit trail entries, policy violations, and regulatory assessment results are simulated for demonstration purposes. No real companies, individuals, financial data, or AI interactions are represented. The regulatory mappings reference real laws and frameworks (FTC guidance, Colorado AI Act, CCPA, etc.) but their application to the demo scenario is illustrative.

---

## Related Projects

- **[MROps](https://pmcavallo.github.io/MROps/)** — 3-agent LangGraph pipeline for model risk validation intake. Authorization boundary assessment for agentic AI systems.
- **[AutoDoc AI](https://pmcavallo.github.io/AutoDoc-AI/)** — 4-agent RAG system for regulatory documentation. 100% source fidelity with full citation traceability.
- **[EvalOps](https://pmcavallo.github.io/evalops/)** — LLM evaluation and observability platform. 285 tests, drift detection, deployed to AWS.
- **[CreditNLP](https://pmcavallo.github.io/creditnlp/)** — Fine-tuned Mistral-7B for credit risk classification. 93.9% accuracy vs. 60% few-shot baseline.
- **[MCP Banking Workflows](https://pmcavallo.github.io/mcp-project/)** — MCP server with 10 tools for AI-assisted model risk management automation.
