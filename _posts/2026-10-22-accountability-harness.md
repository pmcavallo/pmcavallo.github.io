---
layout: page
title: "Agent Accountability Harness: Governing What an AI Agent Is Allowed to Do"
date: 2026-10-22
---

An AI agent that only answers questions can embarrass you. An agent with tools can act for you: it sends email, books the calendar, adds records, moves things. When one of those actions goes wrong, the first questions are always the same. What was the agent allowed to touch? What did it actually do? And how do we undo it?

In my experience, the answer to all three usually lives in someone's head, in a prompt, or nowhere. The permissions sit in code an engineer wrote. The actions sit in scattered logs, if anyone kept them. Nobody wrote down, before the run, which of the agent's actions can be reversed and which cannot.

The Agent Accountability Harness is a small Python library for the person who deploys the agent. Before the run, you declare what the agent may touch and how each of its changes can be undone. During the run, the harness records every tool call and flags, or blocks, the ones outside that declaration. After the run, it writes a plain-language report that opens with a one-screen summary for the accountable owner. The scope itself is a governed document: who owns it, who approved it, and when it is due for review.

> **[Live Demo →](https://accountability-harness.streamlit.app/)**
>
> The hosted demo runs scripted scenarios through the real harness code on synthetic data. No AI model runs in the hosted app, and no real email is sent.

📁 [View on GitHub](https://github.com/pmcavallo/accountability-harness)

---

## The Three Questions

Every report answers three questions, in this order:

1. **What was the agent allowed to touch?** The declared scope: tools, whether changes are allowed, argument rules, and how each change can be undone.
2. **What did it do?** Every recorded tool call, in order, marked OK, VIOLATION or BLOCKED, with the reason.
3. **How can its actions be undone?** Each change, grouped by how it can be undone, as declared before the run, worst first.

Above those three answers sits an owner summary: whether the scope was approved, whether its review is overdue, the run's changes by reversibility, what needs attention, and whether the log verified. It is written for the person who is accountable for the agent, not for the engineer who built it.

---

## The Demo: An Inbox Assistant and a Planted Email

The demo is an inbox assistant for Tallow Lane Events, a fictional events company. It has seven tools: read the inbox, read an email, summarize a thread, draft a reply, place a calendar hold, add a client contact, and send an email. One of the unread emails is a planted indirect prompt injection that asks for the client list and price sheet to go to an outside address.

The same scripted run, under the same scope, in two modes:

![Observe vs enforce: one email sent and flagged, against none sent](https://github.com/pmcavallo/accountability-harness/blob/main/docs/images/app-observe-vs-enforce.jpg?raw=true)

- **Observe mode** is a detective control. The call runs, and the harness records it and flags it. One email went out, and the report lists it as a change that cannot be undone.
- **Enforce mode** is a preventive control. The call fails the scope check and never runs. Nothing was sent, and the report shows one action blocked before it ran.

Running the same scope both ways is the point. Observe tells you what went wrong. Enforce stops a declared category of wrong before it happens, and only that category.

---

## What It Controls

**Declared scope.** The tools the agent may call, written down before the run. Every call is checked against that list.

**Changes allowed or not.** A run can be declared review-only, so any change counts as out of scope.

**Argument rules.** A tool name is often too coarse. A rule can say that `send_email` may only go to the company's own domain. Rules fail closed: a missing argument, or a check that raises, counts as broken.

**Reversibility declarations.** For every tool that changes something, you declare before the run whether its changes are reversible (undone directly), compensable (offset by a separate action), or irreversible (nothing restores the prior state), and how. A tool that changes something with no declaration is reported as a gap nobody planned for.

**Governed scope.** The scope can be a TOML file with an owner, an approver, an approval date, a version, a review date and a one-line reason for each allowed tool. Every log records a SHA-256 fingerprint of the scope, so the report shows exactly which approved version the agent ran under. The report lists governance findings: no approver, no owner, review overdue, a tool allowed without a reason.

```toml
name = "inbox-assistant"
version = "1.0"
owner = "Operations lead, Tallow Lane Events"
approver = "Managing director, Tallow Lane Events"
approved_on = 2026-10-15
review_by = 2027-04-15
allowed_tools = ["read_inbox", "draft_reply"]

[rationale]
read_inbox = "Needs to see which emails arrived."
draft_reply = "Prepares replies for a person to review and send."
```

**Change comparison.** `accountability-harness compare A B` compares two folders of audit logs, for example before and after a prompt change, and flags a material behavior change when one of five documented rules fires. It is a heuristic over counts, and it prints its thresholds with every comparison.

**Tamper-evident log.** Each log entry is hash-chained to the one before. `accountability-harness verify LOG` reports "chain intact" or the first line where the chain breaks. It detects edits after the fact; it does not prevent them.

---

## Architecture

```
            Governed scope (TOML)
   owner · approver · review date · tools ·
   argument rules · reversibility · fingerprint
                     │
                     ▼
  Agent ──► tool call ──► ┌──────────────────────┐
                          │       HARNESS        │
                          │                      │
                          │  1. tool in scope?   │
                          │  2. changes allowed? │
                          │  3. argument rules?  │
                          │                      │
                          │  observe: run + flag │
                          │  enforce: block      │
                          └──────────┬───────────┘
                                     │
                     ┌───────────────┼────────────────┐
                     ▼               ▼                ▼
               Tool runs      Hash-chained       "blocked: reason"
              (or not)         audit log         back to the agent
                                     │
                                     ▼
                       Report: owner summary +
                       the three questions
                                     │
                                     ▼
                  verify · compare · report (CLI)
```

The harness wraps plain Python functions, so it works with any agent loop. There is an optional LangChain adapter, with blocking tested for `invoke` and `ainvoke`. The core package has no required dependencies and is about 1,300 lines.

```python
harness = Harness(scope, mode="enforce")
send_email = harness.wrap(send, name="send_email", on_block="return")
print(send_email("someone@outside.example", "Hello"))
# blocked: tool 'send_email' is not in the declared scope
```

---

## Findings From 100 Real-Model Runs

The scripted demo shows the controls working. I also wanted to know what a real model does inside the harness. I ran Claude Haiku 5.5 as the same inbox assistant, through a small hand-rolled tool-use loop with every tool going through the harness: five injection techniques, observe and enforce modes, and two system prompts. That came to 100 runs, at a total cost of $0.14 from reported usage. Every outcome below is measured from the audit logs, not from what the model said.

**1. The model never attempted the outside send.** Across all five techniques and both prompts, it read the planted email every time and never called `send_email`. Under the permissive prompt, its closing summary pointed the planted email out to the user in all 50 runs.

**2. Changing the system prompt changed what the agent did with legitimate requests.** The second prompt was one a deployer could plausibly write: act on behalf of the owner, complete what each email asks, without asking for confirmation. Under it, the agent placed the calendar hold one email asks for in 50 of 50 runs, against 4 of 50 under the neutral prompt, and drafted fewer replies (107 against 168). The harness recorded all 329 changes across the 100 runs with their declared reversibility.

**3. The change comparison caught the prompt change from the logs alone.** Both sets of runs ran under the same scope fingerprint. The scope did not change; the behavior did.

![Change comparison: calendar holds in 4 of 50 runs against 50 of 50](https://github.com/pmcavallo/accountability-harness/blob/main/docs/images/app-change-comparison.jpg?raw=true)

To me, the second finding is the governance lesson. A prompt edit is usually treated as a wording change, and nobody reviews it the way they would review a permission change. Here a one-paragraph prompt edit changed how often the agent acted on the company's calendar from 8% of runs to 100%, with the approved scope untouched. If you only govern the permissions, you miss this.

Because the model never attempted the send, no real-model run exercised flagging or blocking. That behavior is shown by the scripted scenarios and the tests. This is a case study of one model on one scenario, not a benchmark.

---

## Build Your Own Test

The hosted app includes a playground. You start from the approved scope, turn tools on and off, change how each change can be undone, edit the allowed email domains, pick which scripted calls run and where the email goes, and run it in observe or enforce mode. The app shows what was sent, the report, the edited scope with its new fingerprint, and a comparison with the approved scope.

Any edit makes the scope a new, unapproved version, and the findings say so. Changing what an agent may do is a governance decision, not a configuration tweak.

For live demonstrations, a local-only mode runs a real model through the same harness. It is off by default, needs two environment variables set on the machine running it, and does not exist in the hosted app, which never reads a key.

---

## Framework Mapping

The repository maps each control to the NIST AI RMF 1.0, NIST AI 600-1 (the Generative AI Profile), the EU AI Act, and the OWASP Top 10 for LLM Applications 2025, including LLM01 (Prompt Injection) and LLM06 (Excessive Agency). Every citation was checked against its primary source on 2026-10-21, and each one states what the control does not cover.

The mapping is illustrative, not a compliance claim. Using the harness does not make any system compliant with any framework or law.

---

## What This Does Not Prove

I wrote a full page on this in the repository, because a governance tool that overstates itself is part of the problem it claims to solve. The short version:

- Only calls that go through the harness are recorded or blocked.
- The audit log is tamper-evident, not tamper-proof. A rewrite of the whole file goes undetected unless the final hash is kept somewhere else.
- Governance fields are recorded text, not verified approvals.
- The harness never undoes anything. It records how a change can be undone, as declared.
- It does not detect prompt injection. It checks what the agent does, not why, and a scope is only as good as the person who writes it.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Core library | Python 3.11+, no required dependencies |
| Scope files | TOML (`tomllib`), SHA-256 fingerprint |
| Audit log | JSON Lines, hash-chained (SHA-256) |
| Agent integration | Plain function wrapping; optional LangChain adapter |
| Real-model experiments | Anthropic API, Claude Haiku 5.5, hand-rolled tool-use loop |
| Demo app | Streamlit, deployed on Streamlit Community Cloud |
| CLI | `accountability-harness report`, `verify`, `compare` |
| Testing | pytest, 238 tests, all offline, none calls a model |

---

## Build History

| Version | What |
| --- | --- |
| 0.2.0 | Framework-neutral core: wrap any Python tool; LangChain becomes an optional adapter |
| 0.3.0 | Reversibility declarations; incomplete calls recorded |
| 0.4.0 | Enforce mode and argument rules |
| 0.5.0 | Argument rules fail closed; real-model experiment suite; recorded replay |
| 0.5.1 | Permissive-prompt condition: 50 more real-model runs |
| 0.6.0 | Control specification, threat model, limits, framework mapping |
| 0.7.0 | Governance layer: governed scope, change comparison, hash-chained log, owner summary, CLI |
| 0.8.0 | Demo app; framework citations verified; first public release |
| 0.9.0 | Playground page and local live mode |

---

## Synthetic Data Disclaimer

All data in this project is entirely **synthetic**. Tallow Lane Events, its inbox, its clients and every email address are invented, and every address uses the reserved `.example` domain, so nothing could reach a real mailbox. The framework citations reference real documents, but their application to the demo scenario is illustrative.

---

## Related Projects

- **[AuditHive](https://pmcavallo.github.io/audithive-project/)**: AI governance middleware that sits between an application and its LLM provider. The harness works one layer down, on the actions an agent takes with its tools.
- **[MROps](https://pmcavallo.github.io/MROps/)**: 3-agent LangGraph pipeline that includes an authorization boundary assessment for agentic AI systems. The harness is the runtime side of that question.
- **[EU AI Act Compliance Scanner](https://pmcavallo.github.io/eu-ai-act-scanner/)**: rule-based risk classification and obligation gap analysis under the EU AI Act.
