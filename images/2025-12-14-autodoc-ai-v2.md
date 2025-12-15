---
layout: post
title: "AutoDoc AI v2: Memory-Enhanced Multi-Agentic Documentation System"
date: 2025-12-15
---

AutoDoc AI v2 transforms the original multi-agent system into a truly **multi-agentic** architecture where agents learn from past generations, adapt to different insurance portfolios, and make autonomous routing decisions. The upgrade introduces three critical capabilities: a **3-tier memory system** that enables cross-session learning and user personalization, **LangGraph state machine orchestration** with explicit conditional routing and revision cycles, and **dynamic portfolio detection** that automatically configures agents for Personal Auto, Homeowners, Workers' Compensation, or Commercial Auto documentation. These additions solve the fundamental limitation of v1: every document followed the same fixed pipeline regardless of portfolio complexity or historical patterns. Now, a Workers' Comp model automatically triggers strict compliance checking with 4 revision cycles, while the system learns that "tail development documentation" fails 73% of Workers' Comp reviews and proactively flags it before generation begins.

---

## What's New in v2

Version 1 was **multi-agent**: multiple specialized agents working together in a fixed pipeline. Version 2 is **multi-agentic**: agents that learn, adapt, and make autonomous decisions about workflow routing.

![autodoc aiv2](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/v1VSv2.png?raw=true)

**The Core Insight**: Multi-agent systems execute workflows. Multi-agentic systems decide which workflows to execute, learn from outcomes, and adapt over time.

---

## Architecture Evolution

### v1 Architecture (Fixed Pipeline)

```
PPT Input → Research → Write → Compliance → Editorial → Output
                                    ↓          ↓
                              (feedback loop via Python for-loop)
```

Every document, regardless of portfolio or complexity, followed the identical path.

### v2 Architecture (Adaptive State Machine)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY LAYER                                    │
│  ┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐ │
│  │ Session Memory  │  │ Cross-Session Memory │  │     User Memory         │ │
│  │ (In-Memory)     │  │ (SQLite)             │  │     (JSON)              │ │
│  │                 │  │                      │  │                         │ │
│  │ • Portfolio     │  │ • Issue patterns     │  │ • Tone preference       │ │
│  │ • Metrics used  │  │ • Quality baselines  │  │ • Detail level          │ │
│  │ • Issues found  │  │ • Success patterns   │  │ • Custom rules          │ │
│  └────────┬────────┘  └──────────┬───────────┘  └────────────┬────────────┘ │
│           │                      │                           │              │
│           └──────────────────────┴───────────────────────────┘              │
│                                  │                                          │
│                                  ▼                                          │
│                        ┌─────────────────┐                                  │
│                        │ Memory Manager  │                                  │
│                        └────────┬────────┘                                  │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH ORCHESTRATOR                               │
│                                                                              │
│  ┌────────────────┐      ┌───────────┐      ┌──────────┐      ┌─────────┐  │
│  │detect_portfolio│─────▶│ configure │─────▶│ research │─────▶│  write  │  │
│  └────────────────┘      └───────────┘      └──────────┘      └────┬────┘  │
│         │                      │                                    │       │
│         │ Detects:             │ Applies:                           │       │
│         │ • workers_comp       │ • 10 sections                      ▼       │
│         │ • confidence: 0.85   │ • threshold: 7.5          ┌────────────┐   │
│         │ • keywords: NCCI,    │ • strictness: strict      │ compliance │◀─┐│
│         │   payroll, medical   │ • max_iter: 4             └─────┬──────┘  ││
│         │                      │                                 │         ││
│                                                            ┌─────┴─────┐   ││
│                                                            ▼           ▼   ││
│                                                        (passed)    (failed)││
│                                                            │           │   ││
│                                                            ▼           │   ││
│                                                     ┌───────────┐      │   ││
│                                                     │ editorial │      │   ││
│                                                     └─────┬─────┘      │   ││
│                                                           │            │   ││
│                                                     ┌─────┴─────┐      │   ││
│                                                     ▼           ▼      │   ││
│                                                 (passed)    (failed)   │   ││
│                                                     │           │      │   ││
│                                                     ▼           ▼      ▼   ││
│                                                ┌────────┐  ┌──────────┐   ││
│                                                │complete│  │ revision │───┘│
│                                                └────────┘  └──────────┘    │
│                                                     │                       │
│                                                     ▼                       │
│                                                   [END]                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

The revision node uses `WriterAgent.revise_section()` with collected feedback from both compliance and editorial phases, then routes back to compliance for re-validation. This creates explicit cycles that LangGraph manages as a proper state machine.

---

## Memory System

The memory system enables AutoDoc AI to learn from every document it generates, remember user preferences, and maintain consistency within generation runs.

### The Three Tiers

**Tier 1: Session Memory (In-Memory)**

Tracks state within a single document generation for internal consistency.

```python
from memory.session_memory import SessionMemory

session = SessionMemory()

# Detect portfolio from source content
portfolio = session.detect_portfolio(source_content)
# Returns: "workers_comp" with confidence 0.85

# Record metrics as they're written (prevents contradictions)
session.record_metric("model_performance", "R² = 0.72")
session.record_metric("sample_size", "450,000 policies")

# Track compliance issues for revision
session.record_compliance_issue(
    description="Tail development factors not documented",
    severity="HIGH",
    section="Loss Development Analysis"
)

# Writer agent queries: "What R² did I already write?"
metrics = session.get_recorded_metrics()
# Returns: {"model_performance": "R² = 0.72", "sample_size": "450,000 policies"}
```

**Why it matters**: Without session memory, the same model could be described as having "R² = 0.72" in the Executive Summary and "R² = 0.68" in Model Results. Session memory ensures every agent sees what previous agents wrote.

**Tier 2: Cross-Session Memory (SQLite)**

Learns patterns across all document generations for proactive quality improvement.

```python
from memory.cross_session_memory import CrossSessionMemory

cross_session = CrossSessionMemory(db_path="data/memory/cross_session.db")

# After each generation, record outcomes
cross_session.record_generation(
    session_id="session_abc123",
    portfolio="workers_comp",
    success=True,
    quality_score=8.2,
    iterations=2,
    compliance_issues=["Fee schedule documentation incomplete"],
    sections_revised=["Medical Cost Trends", "Loss Development Analysis"]
)

# Before next Workers' Comp generation, query historical patterns
common_issues = cross_session.get_common_issues("workers_comp", top_n=5)
# Returns:
# [
#     {"issue": "Tail development inadequate", "frequency": 0.73},
#     {"issue": "Fee schedule not addressed", "frequency": 0.65},
#     {"issue": "NCCI mapping incomplete", "frequency": 0.42},
#     {"issue": "Medical trend assumption missing", "frequency": 0.38},
#     {"issue": "Experience mod calculation unclear", "frequency": 0.31}
# ]

# Get quality baseline for comparison
baseline = cross_session.get_quality_baseline("workers_comp")
# Returns: {"avg_score": 7.8, "avg_iterations": 2.3, "total_generations": 12}

# Query successful patterns
patterns = cross_session.get_successful_patterns("workers_comp")
# Returns sections and approaches that consistently score 8.0+
```

**Why it matters**: The compliance agent now knows that 73% of Workers' Comp documents fail on "tail development" before generation starts. It can instruct the writer agent to prioritize this section.

**Tier 3: User Memory (JSON)**

Stores user preferences for personalized generation.

```python
from memory.user_memory import UserMemory

user = UserMemory(user_id="analyst_sarah")

# Set preferences
user.set_preference("tone", "technical")  # vs "executive" or "regulatory"
user.set_preference("detail_level", "high")
user.set_preference("include_code_snippets", True)
user.set_preference("max_section_length", 1500)  # words

# Add custom compliance rules
user.add_custom_rule("Always verify CECL alignment for loss reserving models")
user.add_custom_rule("Include state-specific fee schedule references")

# Exclude sections user doesn't need
user.set_preference("skip_sections", ["Business Context"])

# Writer agent retrieves preferences
instructions = user.get_writing_instructions()
# Returns formatted string:
# "TONE: technical
#  DETAIL: high
#  Include code snippets where relevant
#  Maximum section length: 1500 words
#  CUSTOM RULES:
#    - Always verify CECL alignment for loss reserving models
#    - Include state-specific fee schedule references"
```

**Why it matters**: An executive reviewing models wants 400-word summaries. A technical validator wants 1500-word deep dives with code. User memory ensures the same system serves both without manual configuration each time.

### Memory Manager (Unified Interface)

```python
from memory.memory_manager import MemoryManager

# Initialize with user context
memory = MemoryManager(user_id="analyst_sarah", db_path="data/memory/cross_session.db")

# Start session (detects portfolio, loads historical patterns)
session_id = memory.start_session(source_content)
print(f"Portfolio: {memory.session.detected_portfolio}")
print(f"Historical issues to watch: {memory.get_proactive_warnings()}")

# Get context for each agent phase
research_context = memory.get_research_context("Methodology")
# Returns: RAG focus areas + successful patterns from past docs

writing_context = memory.get_writing_context("Validation")
# Returns: Metrics already written + user tone preferences + section patterns

compliance_context = memory.get_compliance_context()
# Returns: Portfolio checkpoints + custom rules + common historical issues

editorial_context = memory.get_editorial_context()
# Returns: Quality baseline + user preferences + issues already flagged

# End session (persists learnings)
memory.end_session(final_score=8.2, success=True)
# Writes to cross_session.db for future generations
```

### Memory Integration in Orchestrator

The memory manager is wired into every phase of the LangGraph workflow:

```python
# From agents/graph_nodes.py

def research_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    memory = state.get("memory_manager")
    
    for section in sections:
        # Memory provides historical context
        context = memory.get_research_context(section)
        
        # Includes:
        # - "Past workers_comp docs prioritized NCCI classification"
        # - "Successful Methodology sections averaged 1,200 words"
        # - "User prefers technical detail with code snippets"
        
        findings = research_agent.research_topic(
            topic=f"{section} {model_type}",
            additional_context=context  # Memory-enhanced
        )
```

---

## LangGraph Implementation

The orchestration is built as an explicit state machine using LangGraph, making the workflow's conditional logic visible and debuggable.

### State Definition

```python
# From agents/graph_state.py

from typing import TypedDict, List, Dict, Optional

class DocumentState(TypedDict, total=False):
    # === Input ===
    document_title: str
    source_content: str
    user_id: str
    
    # === Portfolio Detection ===
    detected_portfolio: str      # personal_auto, homeowners, workers_comp, commercial_auto
    detected_model_type: str     # frequency, severity
    portfolio_confidence: float  # 0.0 to 1.0
    detection_keywords: List[str]
    
    # === Dynamic Configuration ===
    required_sections: List[str]
    quality_threshold: float     # 7.0 for personal_auto, 7.5 for others
    max_iterations: int          # 3 standard, 4 for workers_comp
    compliance_strictness: str   # standard, elevated, strict
    custom_instructions: str     # Built from portfolio config
    
    # === Generation State ===
    current_document: str
    sections_written: List[Dict]
    research_contexts: Dict[str, str]
    
    # === Quality Control ===
    compliance_passed: bool
    editorial_passed: bool
    quality_score: float
    compliance_issues: List[Dict]
    editorial_issues: List[Dict]
    
    # === Iteration Control ===
    current_iteration: int
    revision_history: List[Dict]
    
    # === Output ===
    final_document: Optional[str]
    generation_successful: bool
```

### Node Functions

Each node is a pure function that receives state and returns partial updates:

```python
# From agents/graph_nodes.py

def detect_portfolio(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze source content to determine insurance portfolio."""
    
    source_content = state.get("source_content", "").lower()
    
    PORTFOLIO_KEYWORDS = {
        "personal_auto": [
            "driver age", "vehicle type", "collision", "territory",
            "prior claims", "comprehensive", "liability", "pip"
        ],
        "homeowners": [
            "dwelling", "protection class", "cat ", "hurricane",
            "roof", "coverage a", "demand surge", "coastal"
        ],
        "workers_comp": [
            "ncci", "payroll", "medical cost", "indemnity",
            "classification code", "experience mod", "fee schedule",
            "tail development", "monopolistic state"
        ],
        "commercial_auto": [
            "fleet", "gvw", "social inflation", "nuclear verdict",
            "driver turnover", "trucking", "mcs-90", "jurisdiction"
        ]
    }
    
    # Count keyword matches
    match_counts = {}
    for portfolio, keywords in PORTFOLIO_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in source_content]
        match_counts[portfolio] = len(matches)
    
    best_portfolio = max(match_counts, key=match_counts.get)
    confidence = min(1.0, match_counts[best_portfolio] / 10.0)
    
    # Require minimum confidence
    if match_counts[best_portfolio] < 3:
        best_portfolio = "personal_auto"
        confidence = 0.3
    
    return {
        "detected_portfolio": best_portfolio,
        "portfolio_confidence": confidence,
        "detection_keywords": [kw for kw in PORTFOLIO_KEYWORDS[best_portfolio] 
                              if kw in source_content][:10]
    }


def configure_for_portfolio(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load portfolio-specific configuration."""
    
    portfolio = state.get("detected_portfolio")
    config = PORTFOLIO_REGISTRY.get_config(portfolio)
    
    # Build custom instructions
    instructions = [
        f"PORTFOLIO: {config.display_name}",
        f"EXPOSURE BASIS: {config.exposure_basis}",
        "",
        "FOCUS AREAS:",
        *[f"  - {area}" for area in config.research_focus_areas],
        "",
        "COMMON PITFALLS TO AVOID:",
        *[f"  - {pitfall}" for pitfall in config.common_pitfalls],
        "",
        "COMPLIANCE CHECKPOINTS:",
        *[f"  - {check}" for check in config.compliance_checkpoints]
    ]
    
    return {
        "required_sections": config.required_sections + config.portfolio_specific_sections,
        "quality_threshold": config.min_quality_score,
        "max_iterations": config.max_iterations,
        "compliance_strictness": config.compliance_strictness,
        "custom_instructions": "\n".join(instructions)
    }


def revision_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    """Revise sections based on compliance and editorial feedback."""
    
    compliance_issues = state.get("compliance_issues", [])
    editorial_issues = state.get("editorial_issues", [])
    
    # Build revision instructions from feedback
    revision_instructions = []
    
    for issue in compliance_issues:
        if issue.get("severity") in ["CRITICAL", "HIGH"]:
            revision_instructions.append(f"COMPLIANCE: {issue['description']}")
    
    for issue in editorial_issues:
        if issue.get("priority") in ["CRITICAL", "HIGH"]:
            revision_instructions.append(f"EDITORIAL: {issue['description']}")
    
    # Revise each section with WriterAgent
    writer = WriterAgent()
    revised_sections = []
    
    for section in state.get("sections_written", []):
        revised = writer.revise_section(
            original_content=section,
            revision_instructions="\n".join(revision_instructions),
            source_content=state.get("source_content")
        )
        revised_sections.append(revised)
    
    return {
        "sections_written": revised_sections,
        "current_iteration": state.get("current_iteration", 0) + 1,
        "revision_history": state.get("revision_history", []) + [{
            "iteration": state.get("current_iteration"),
            "issues_addressed": len(revision_instructions)
        }]
    }
```

### Router Functions

Routers determine which edge to take based on current state:

```python
# From agents/graph_nodes.py

def route_after_compliance(state: Dict[str, Any]) -> str:
    """Decide next step after compliance check."""
    
    compliance_passed = state.get("compliance_passed", False)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if compliance_passed:
        return "editorial"
    elif current_iteration < max_iterations:
        return "revision"
    else:
        return "complete"  # Accept what we have


def route_after_editorial(state: Dict[str, Any]) -> str:
    """Decide next step after editorial review."""
    
    editorial_passed = state.get("editorial_passed", False)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if editorial_passed:
        return "complete"
    elif current_iteration < max_iterations:
        return "revision"
    else:
        return "complete"


def route_after_revision(state: Dict[str, Any]) -> str:
    """After revision, always loop back to compliance."""
    return "compliance"
```

### Graph Assembly

```python
# From agents/langgraph_orchestrator.py

from langgraph.graph import StateGraph, END

def create_workflow():
    """Assemble the documentation generation graph."""
    
    workflow = StateGraph(DocumentState)
    
    # Add nodes
    workflow.add_node("detect_portfolio", detect_portfolio)
    workflow.add_node("configure", configure_for_portfolio)
    workflow.add_node("research", research_phase)
    workflow.add_node("write", writing_phase)
    workflow.add_node("compliance", compliance_phase)
    workflow.add_node("editorial", editorial_phase)
    workflow.add_node("revision", revision_phase)
    workflow.add_node("complete", complete_workflow)
    
    # Linear edges
    workflow.add_edge("detect_portfolio", "configure")
    workflow.add_edge("configure", "research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "compliance")
    
    # Conditional edges (this is where cycles are defined)
    workflow.add_conditional_edges(
        "compliance",
        route_after_compliance,
        {
            "editorial": "editorial",
            "revision": "revision",
            "complete": "complete"
        }
    )
    
    workflow.add_conditional_edges(
        "editorial",
        route_after_editorial,
        {
            "complete": "complete",
            "revision": "revision"
        }
    )
    
    workflow.add_conditional_edges(
        "revision",
        route_after_revision,
        {
            "compliance": "compliance"  # Loop back
        }
    )
    
    # Entry and exit
    workflow.set_entry_point("detect_portfolio")
    workflow.add_edge("complete", END)
    
    return workflow.compile()
```

### Usage

```python
from agents.langgraph_orchestrator import LangGraphOrchestrator

# Initialize
orchestrator = LangGraphOrchestrator(user_id="analyst_sarah")

# Generate documentation
document, state = orchestrator.generate_documentation(
    document_title="2024 WC Frequency Model",
    document_type="model_doc",
    source_content=ppt_content
)

# Inspect results
print(f"Portfolio: {state['detected_portfolio']}")           # workers_comp
print(f"Confidence: {state['portfolio_confidence']:.0%}")    # 85%
print(f"Quality Score: {state['quality_score']:.1f}/10")     # 8.2/10
print(f"Iterations: {state['current_iteration']}")           # 2
print(f"Sections: {len(state['required_sections'])}")        # 10
print(f"Success: {state['generation_successful']}")          # True
```

---

## Audit Trail for Regulated Industries

In regulated industries, "it worked" isn't enough. Auditors ask: "What path did this document take? What decisions were made? Can you prove the process was followed?"

LangGraph's checkpointing captures every state transition automatically. No manual logging. No parsing stdout. The graph records its own execution.

### Execution History

After every generation, query what happened:

```python
orchestrator = LangGraphOrchestrator(enable_checkpointing=True)  # Default: ON

document, state = orchestrator.generate_documentation(
    document_title="2024 WC Frequency Model",
    source_content=ppt_content
)

# Get execution history
history = orchestrator.get_execution_history()
for entry in history:
    print(f"Step {entry.step}: {entry.node}")
    print(f"  Iteration: {entry.iteration}")
    print(f"  Compliance Passed: {entry.compliance_passed}")
```

Output:
```
Step 0: detect_portfolio
  Iteration: None
  Compliance Passed: None

Step 1: configure
  Iteration: None
  Compliance Passed: None

Step 2: research
  Iteration: 1
  Compliance Passed: None

Step 3: write
  Iteration: 1
  Compliance Passed: None

Step 4: compliance
  Iteration: 1
  Compliance Passed: False

Step 5: revision
  Iteration: 1
  Compliance Passed: False

Step 6: compliance
  Iteration: 2
  Compliance Passed: True

Step 7: editorial
  Iteration: 2
  Compliance Passed: True

Step 8: complete
  Iteration: 2
  Compliance Passed: True
```

### Full Audit Log

For regulatory review, get the complete audit log:

```python
audit_log = orchestrator.get_audit_log()

print(f"Thread ID: {audit_log.thread_id}")
print(f"Portfolio: {audit_log.detected_portfolio}")
print(f"Execution Path: {audit_log.get_path_summary()}")
print(f"Total Iterations: {audit_log.total_iterations}")
print(f"Final Quality: {audit_log.final_quality_score}/10")
print(f"Success: {audit_log.generation_successful}")

# Get decision points (where routing happened)
for decision in audit_log.get_decision_points():
    print(f"{decision['node']}: {'PASSED' if decision['passed'] else 'FAILED'} -> {decision['routed_to']}")
```

Output:
```
Thread ID: abc123-def456
Portfolio: workers_comp
Execution Path: detect_portfolio -> configure -> research -> write -> compliance -> revision -> compliance -> editorial -> complete
Total Iterations: 2
Final Quality: 8.2/10
Success: True

compliance: FAILED -> revision
compliance: PASSED -> editorial
editorial: PASSED -> complete
```

### Streamlit UI Integration

The `get_execution_summary()` method returns data formatted for display:

```python
summary = orchestrator.get_execution_summary()

# Ready for Streamlit
st.subheader("Execution Audit Trail")
st.write(f"**Portfolio Detected:** {summary['portfolio_detected']}")
st.write(f"**Execution Path:** {summary['execution_path']}")
st.write(f"**Quality Score:** {summary['final_quality_score']}")

st.subheader("Decision Points")
for decision in summary['decision_points']:
    status = "✅ PASSED" if decision['passed'] else "❌ FAILED"
    st.write(f"**{decision['node']}**: {status} -> routed to {decision['routed_to']}")
```

### Console Trace for Debugging

```python
orchestrator.print_execution_trace()
```

Output:
```
======================================================================
EXECUTION AUDIT LOG
======================================================================
Thread ID:    abc123-def456
Document:     2024 WC Frequency Model
Portfolio:    workers_comp
Start Time:   2024-12-15T10:30:00
End Time:     2024-12-15T10:42:00
Success:      True
Quality:      8.2/10
Iterations:   2

----------------------------------------------------------------------
EXECUTION PATH
----------------------------------------------------------------------
detect_portfolio -> configure -> research -> write -> compliance -> revision -> compliance -> editorial -> complete

----------------------------------------------------------------------
STATE TRANSITIONS
----------------------------------------------------------------------
Step  Node                Iteration   Phase
----------------------------------------------------------------------
0     detect_portfolio    -           -
1     configure           -           -
2     research            1           research
3     write               1           generation
4     compliance          1           quality_control
5     revision            1           quality_control
6     compliance          2           quality_control
7     editorial           2           quality_control
8     complete            2           complete

----------------------------------------------------------------------
DECISION POINTS
----------------------------------------------------------------------
Step 4: compliance -> FAILED -> routed to revision
Step 6: compliance -> PASSED -> routed to editorial
Step 7: editorial -> PASSED -> routed to complete
======================================================================
```

### Why This Matters for Model Risk Management

| Audit Question | Without Audit Trail | With Audit Trail |
|----------------|---------------------|------------------|
| "What path did document X take?" | Parse logs, add print statements | `audit_log.get_path_summary()` |
| "Why did it revise twice?" | Read code, trace logic | `audit_log.get_decision_points()` |
| "Can you prove compliance was checked?" | Manual documentation | Automatic state history |
| "What was the quality score at each step?" | Not captured | `entry.quality_score` at each node |
| "Can you reproduce this exact run?" | Start over, hope for same result | `app.get_state(config)` returns exact state |

This isn't just debugging. This is SR 11-7 compliance. The audit trail proves the process was followed, not just that the output exists.

---

## Portfolio Detection & Dynamic Routing

The system automatically detects the insurance portfolio from source content and applies portfolio-specific configurations.

### Portfolio Configurations

| Portfolio | Quality Threshold | Max Iterations | Compliance | Unique Sections |
|-----------|-------------------|----------------|------------|-----------------|
| **Personal Auto** | 7.0 | 3 | Standard | (baseline) |
| **Homeowners** | 7.5 | 3 | Elevated | CAT Model Integration, Demand Surge Analysis |
| **Workers' Comp** | 7.5 | 4 | Strict | Loss Development Analysis, Medical Cost Trends, NCCI Mapping |
| **Commercial Auto** | 7.5 | 3 | Elevated | Fleet Risk Analysis, Social Inflation Trends, Jurisdiction Tiers |

### Detection Keywords

```python
PORTFOLIO_KEYWORDS = {
    "personal_auto": [
        "driver age", "vehicle type", "collision", "bodily injury", "territory",
        "prior claims", "comprehensive", "liability", "pip", "uninsured motorist",
        "deductible", "annual mileage", "good driver", "multi-car", "garaging"
    ],
    "homeowners": [
        "dwelling", "protection class", "cat ", "hurricane", "hail", "roof",
        "construction type", "coverage a", "coverage b", "coverage c",
        "replacement cost", "wind", "tornado", "wildfire", "flood zone",
        "coastal", "air ", "rms", "corelogic", "demand surge"
    ],
    "workers_comp": [
        "injury", "ncci", "payroll", "medical cost", "indemnity", "classification code",
        "experience mod", "body part", "nature of injury", "cause of injury",
        "temporary disability", "permanent disability", "medical only", "lost time",
        "fee schedule", "monopolistic state", "tail development"
    ],
    "commercial_auto": [
        "fleet", "vehicle class", "radius", "gvw", "for-hire", "driver turnover",
        "commercial vehicle", "trucking", "light truck", "medium truck", "heavy truck",
        "tractor", "trailer", "hired auto", "non-owned auto", "mcs-90",
        "social inflation", "nuclear verdict", "jurisdiction"
    ]
}
```

### Routing Example

**Input**: PowerPoint containing "This model uses NCCI classification codes to predict injury frequency based on payroll exposure. Medical cost trends and indemnity duration factors are incorporated."

**Detection Result**:
- Portfolio: `workers_comp`
- Confidence: `0.85`
- Keywords matched: `["ncci", "injury", "payroll", "medical cost", "indemnity"]`

**Configuration Applied**:
- Quality threshold: `7.5` (stricter than personal auto's 7.0)
- Max iterations: `4` (one extra revision cycle)
- Compliance strictness: `strict`
- Required sections: 10 (includes Loss Development Analysis, Medical Cost Trends)
- Focus areas: NCCI codes, tail factors, fee schedules
- Common pitfalls: "Tail development factors inadequate", "Medical fee schedule updates not incorporated"

---

## Knowledge Base

The RAG corpus was expanded from 1 portfolio (7 documents) to 4 portfolios (23 documents):

| Collection | Documents | Purpose |
|------------|-----------|---------|
| **Anchor Documents** | 4 | Methodology guides per portfolio |
| **Regulations** | 4 | Regulatory compilations per portfolio |
| **Audit Findings** | 4 | Historical audit issues per portfolio |
| **Model Docs** | 11 | Example documentation across portfolios |
| **Total** | **23** | Complete multi-portfolio corpus |

### Corpus Structure

```
data/
├── anchor_document/
│   ├── personal_auto_methodology_guide.md
│   ├── homeowners_methodology_guide.md
│   ├── workers_comp_methodology_guide.md
│   └── commercial_auto_methodology_guide.md
│
├── regulations/
│   ├── personal_auto_regulations.md
│   ├── homeowners_regulations.md
│   ├── workers_comp_regulations.md
│   └── commercial_auto_regulations.md
│
├── audit_findings/
│   ├── personal_auto_audit_findings.md
│   ├── homeowners_audit_findings.md
│   ├── workers_comp_audit_findings.md
│   └── commercial_auto_audit_findings.md
│
├── synthetic_docs/
│   ├── 2022_frequency_model_doc.md
│   ├── 2022_severity_model_doc.md
│   ├── 2023_homeowners_fire_frequency.md
│   ├── 2023_homeowners_wind_hail_severity.md
│   ├── 2023_workers_comp_injury_frequency.md
│   ├── 2023_workers_comp_medical_severity.md
│   ├── 2023_commercial_auto_fleet_frequency.md
│   ├── 2023_commercial_auto_liability_severity.md
│   └── ... (11 total)
│
└── memory/
    ├── cross_session.db              # SQLite for pattern learning
    └── user_preferences/
        └── {user_id}.json            # Per-user preferences
```

---

## Test Coverage

```
========================= test session starts =========================

tests/test_session_memory.py::TestSessionMemory
    test_detect_personal_auto                    PASSED
    test_detect_homeowners                       PASSED
    test_detect_workers_comp                     PASSED
    test_detect_commercial_auto                  PASSED
    test_detect_frequency_model                  PASSED
    test_detect_severity_model                   PASSED
    test_record_and_retrieve_metrics             PASSED
    test_record_compliance_issues                PASSED
    ... (24 tests total)                         24 passed

tests/test_cross_session_memory.py::TestCrossSessionMemory
    test_record_generation                       PASSED
    test_get_common_issues                       PASSED
    test_get_quality_baseline                    PASSED
    test_get_successful_patterns                 PASSED
    ... (16 tests total)                         16 passed

tests/test_user_memory.py::TestUserMemory
    test_set_and_get_preferences                 PASSED
    test_add_custom_rules                        PASSED
    test_persistence_across_sessions             PASSED
    test_get_writing_instructions                PASSED
    ... (24 tests total)                         24 passed

tests/test_memory_manager.py::TestMemoryManager
    test_start_session_detects_portfolio         PASSED
    test_get_research_context                    PASSED
    test_get_compliance_context                  PASSED
    test_end_session_persists                    PASSED
    ... (24 tests total)                         24 passed

tests/test_langgraph_orchestrator.py::TestLangGraph
    test_detect_personal_auto                    PASSED
    test_detect_homeowners                       PASSED
    test_detect_workers_comp                     PASSED
    test_detect_commercial_auto                  PASSED
    test_workers_comp_strict_compliance          PASSED
    test_homeowners_has_cat_section              PASSED
    test_commercial_auto_has_social_inflation    PASSED
    test_compliance_passed_routes_to_editorial   PASSED
    test_compliance_failed_routes_to_revision    PASSED
    test_editorial_passed_routes_to_complete     PASSED
    test_revision_loops_to_compliance            PASSED
    test_max_iterations_exits_gracefully         PASSED
    test_workflow_compiles                       PASSED
    test_end_to_end_portfolio_detection          PASSED
    ... (44 tests total)                         44 passed

========================= 132 passed in 34.93s =========================
```

---

## Project Structure

```
autodoc-ai/
├── agents/
│   ├── orchestrator.py              # v1 orchestrator (backward compat)
│   ├── langgraph_orchestrator.py    # v2 LangGraph state machine
│   ├── graph_state.py               # DocumentState TypedDict
│   ├── graph_nodes.py               # Node functions + routers
│   ├── research_agent.py
│   ├── writer_agent.py
│   ├── compliance_agent.py
│   └── editor_agent.py
│
├── memory/
│   ├── session_memory.py            # Tier 1: In-memory state
│   ├── cross_session_memory.py      # Tier 2: SQLite persistence
│   ├── user_memory.py               # Tier 3: JSON preferences
│   └── memory_manager.py            # Unified interface
│
├── config/
│   └── portfolio_configs.py         # 4 portfolio configurations
│
├── rag/
│   ├── ingestion.py
│   └── retrieval.py
│
├── data/
│   ├── synthetic_docs/              # 11 model documentation files
│   ├── anchor_document/             # 4 methodology guides
│   ├── regulations/                 # 4 regulation compilations
│   ├── audit_findings/              # 4 audit finding catalogs
│   └── memory/
│       ├── cross_session.db         # Pattern learning database
│       └── user_preferences/        # Per-user JSON files
│
├── tests/
│   ├── test_session_memory.py       # 24 tests
│   ├── test_cross_session_memory.py # 16 tests
│   ├── test_user_memory.py          # 24 tests
│   ├── test_memory_manager.py       # 24 tests
│   └── test_langgraph_orchestrator.py # 44 tests
│
└── app/
    └── streamlit_app.py             # Demo interface
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | State machine with conditional routing |
| **LLM** | Claude Sonnet 4 | All agent generation and evaluation |
| **Memory** | SQLite + JSON | Cross-session learning + user preferences |
| **RAG** | ChromaDB | Vector storage and retrieval |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 |
| **Evaluation** | RAGAS + LLM-as-Judge | Quality scoring |
| **Frontend** | Streamlit | Demo interface |
| **Testing** | pytest | 132 tests |

---

## What This Demonstrates

### Multi-Agentic Patterns
- **State Machine Orchestration**: LangGraph with explicit conditional edges and cycles
- **Memory Architecture**: 3-tier system enabling learning and personalization
- **Dynamic Routing**: Portfolio detection drives agent configuration
- **Feedback Loops**: Revision cycles until quality threshold met
- **Audit Trail**: Automatic execution history for regulatory compliance

### Production AI Skills
- **RAG Systems**: Hybrid retrieval with portfolio-specific corpus
- **LLM Evaluation**: RAGAS metrics + LLM-as-Judge for quality
- **Cost Optimization**: Targeted revisions, token tracking
- **Error Handling**: Graceful degradation, max iteration limits
- **Observability**: Full state history at every node transition

### Domain Expertise
- **Insurance Model Risk**: SR 11-7, ASOP compliance
- **Multi-Portfolio**: Personal Auto, Homeowners, Workers' Comp, Commercial Auto
- **Regulatory Knowledge**: NAIC, NCCI, state-specific requirements
- **Audit Readiness**: Execution traces that answer "what path did this take?"

---

## License

MIT License - see LICENSE file for details.

---

*"The difference between multi-agent and multi-agentic is memory. Without memory, agents follow scripts. With memory, agents learn, adapt, and make decisions that improve over time. AutoDoc AI v2 proves that production AI systems need both: the structured orchestration of LangGraph and the adaptive learning of persistent memory."*

---

**v2 completed December 2024**
