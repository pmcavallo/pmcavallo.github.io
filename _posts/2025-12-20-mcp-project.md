---
layout: post
title: "MCP Banking Workflows: AI-Powered Model Risk Management Automation"
date: 2025-12-20
---

MCP Banking Workflows is a production-ready Model Context Protocol (MCP) server that automates model documentation validation, dependency analysis, and regulatory compliance checking for banking credit risk models. The system addresses a critical bottleneck in Model Risk Management: **documentation drift** - where model code evolves through versions while PowerPoint presentations, Excel data dictionaries, and Word white papers fall out of sync. Using 10 specialized tools accessible via Claude or any MCP-compatible LLM, the server enables AI assistants to validate cross-file consistency, analyze change impact across a 16-model dependency graph, check SR 11-7 compliance, and generate analyst onboarding briefs - transforming hours of manual cross-referencing into seconds of automated validation.

---

## The Problem

Model Risk Management teams at banks face a persistent challenge that creates regulatory risk and audit exposure:

**Documentation Drift:**

* Model code evolves through versions (v3.5 → v4.2)
* PowerPoint presentations shown to validation committees contain outdated thresholds
* Excel data dictionaries don't match the variables actually used in SAS code
* White papers describe methodology that was changed two versions ago
* Committee meeting minutes reference superseded parameter values

**Manual Validation Burden:**

* Cross-referencing a single model across SAS code, Excel dictionary, Word documentation, and PowerPoint takes 4-8 hours
* Each model version change triggers documentation updates across 5+ files
* New analysts spend weeks understanding model dependencies and institutional knowledge
* Version comparisons require line-by-line code review

**Regulatory Exposure:**

* Federal Reserve SR 11-7 requires accurate, complete model documentation
* OCC 2011-12 mandates documented model governance processes
* Inadequate documentation can delay or block model approvals
* Audit findings on documentation gaps cost weeks of remediation effort

**Dependency Blindness:**

* Banks maintain portfolios of 10-50+ interconnected models
* Changing one model triggers cascading revalidation requirements
* Data source changes affect multiple models simultaneously
* Impact assessment requires manual tracing through model registries

The problem isn't lack of effort - it's the inherent difficulty of maintaining consistency across file formats, tracking dependencies across model portfolios, and ensuring regulatory compliance at scale.

---

## The Solution

MCP Banking Workflows introduces an AI-accessible toolkit that enables LLMs to perform model risk management tasks that previously required senior analyst expertise. The server exposes 10 tools via the Model Context Protocol, allowing Claude (or any MCP-compatible assistant) to read, compare, and validate model artifacts autonomously.

### Core Architecture

```
                                    MCP Protocol
    Claude/LLM  <──────────────────────────────────>  Banking MCP Server
                                                             │
                         ┌───────────────────────────────────┼───────────────────────────────────┐
                         │                                   │                                   │
                         v                                   v                                   v
                  ┌─────────────┐                    ┌──────────────┐                   ┌──────────────┐
                  │ Model Code  │                    │    Excel     │                   │    Word      │
                  │ (.sas, .py) │                    │ Dictionaries │                   │  Documents   │
                  └─────────────┘                    └──────────────┘                   └──────────────┘
                         │                                   │                                   │
                         │                                   v                                   │
                         │                          ┌──────────────┐                            │
                         └─────────────────────────>│    Model     │<───────────────────────────┘
                                                    │   Registry   │
                                                    │    (JSON)    │
                                                    └──────────────┘
                                                           │
                         ┌─────────────────────────────────┼─────────────────────────────────┐
                         v                                 v                                 v
                  ┌─────────────┐                  ┌──────────────┐                 ┌──────────────┐
                  │ PowerPoint  │                  │  Dependency  │                 │   SR 11-7    │
                  │Presentations│                  │    Graph     │                 │  Compliance  │
                  └─────────────┘                  │   (16 models)│                 │   Checker    │
                                                   └──────────────┘                 └──────────────┘
```

The server employs a **filesystem-first approach**: all model artifacts (SAS code, Excel dictionaries, Word documentation, PowerPoint presentations) remain in their native formats. The MCP server provides read-only access via regex parsing, library extraction (openpyxl, python-docx, python-pptx), and structured JSON output - enabling LLMs to reason about model consistency without modifying source files.

---

## Key Features

### 10 Specialized Tools

| Tool | Description |
|------|-------------|
| `list_model_files` | List all SAS and Python model files in the repository |
| `extract_sas_parameters` | Parse DSCR/LTV thresholds, property segments, and model variables from SAS code |
| `read_excel_dictionary` | Extract variable definitions, data types, and business rules from Excel data dictionaries |
| `check_cross_file_consistency` | Validate that model code matches data dictionary (thresholds, variables, segments) |
| `compare_model_versions` | Diff two model versions to identify changes in variables, thresholds, and performance |
| `list_word_documents` | List available white papers, meeting minutes, and validation reports |
| `extract_word_content` | Query Word documents by section (methodology, limitations, performance, etc.) |
| `map_model_dependencies` | Analyze upstream/downstream dependencies and estimate revalidation effort |
| `check_sr11_compliance` | Validate model documentation against Federal Reserve SR 11-7 requirements |
| `check_presentation_consistency` | Detect outdated information in PowerPoint presentations vs current model code |

### Cross-File Consistency Validation

**Problem Solved:** "Does my SAS code match what's documented in the data dictionary?"

```
Tool: check_cross_file_consistency
Input: sas_filename="CRE_PD_v4.2.sas", excel_filename="CRE_PD_data_dictionary.xlsx"

Output:
======================================================================
CROSS-FILE CONSISTENCY CHECK
======================================================================
SAS File:   CRE_PD_v4.2.sas
Excel File: CRE_PD_data_dictionary.xlsx
======================================================================

1. DSCR THRESHOLD COMPARISON
----------------------------------------------------------------------
   SAS Code:         1.25
   Excel Dictionary: 1.25
   Status: CONSISTENT

2. LTV THRESHOLD COMPARISON
----------------------------------------------------------------------
   SAS Code:         0.80
   Excel Dictionary: 0.80
   Status: CONSISTENT

3. PROPERTY TYPE SEGMENTATION
----------------------------------------------------------------------
   SAS segments found: 4
      - MULTIFAMILY -> segment 1
      - OFFICE -> segment 2
      - RETAIL -> segment 3
      - INDUSTRIAL -> segment 4
   Status: CONSISTENT - 4 property segments defined

4. MODEL VARIABLES
----------------------------------------------------------------------
   Variables in SAS model: DSCR, LTV, property_type, loan_age, unemployment_rate
   Variables documented in Excel: 12
   Status: CONSISTENT - All model variables documented

======================================================================
SUMMARY: 4 of 4 checks passed
Result: ALL CHECKS PASSED
======================================================================
```

### Model Version Comparison

**Problem Solved:** "What changed between v3.5 and v4.2 of the CRE-PD model?"

```
Tool: compare_model_versions
Input: model_name="CRE_PD", version_old="v3.5", version_new="v4.2"

Output:
{
  "comparison": {
    "model_name": "CRE_PD",
    "old_version": "v3.5",
    "new_version": "v4.2"
  },
  "changes": {
    "variables": {
      "added": ["unemployment_rate"],
      "removed": [],
      "unchanged": ["DSCR", "LTV", "property_type", "loan_age"]
    },
    "thresholds": {
      "dscr": {"old": 1.20, "new": 1.25, "changed": true},
      "ltv": {"old": 0.80, "new": 0.80, "changed": false}
    },
    "segments": {
      "old_count": 3,
      "new_count": 4,
      "changed": true,
      "details": "Added INDUSTRIAL as separate segment (previously combined with RETAIL)"
    },
    "performance": {
      "auc_old": 0.823,
      "auc_new": 0.842,
      "improvement": "+2.3%"
    }
  },
  "summary": {
    "total_changes": 3,
    "breaking_changes": 0,
    "enhancements": [
      "Added unemployment_rate variable",
      "DSCR threshold changed from 1.20 to 1.25",
      "Expanded from 3 to 4 property segments",
      "Improved AUC by +2.3%"
    ],
    "recommendation": "v4.2 is a significant enhancement over v3.5 with improved discrimination and new variables"
  }
}
```

### Dependency and Impact Analysis

**Problem Solved:** "If I change the CRE-PD model, what downstream models are affected?"

The server maintains a 16-model dependency graph across 4 layers:

| Layer | Models | Description |
|-------|--------|-------------|
| Layer 1 | CRE-PD, CRE-LGD, CNI-PD, RES-MORT-PD, CC-PD, AUTO-PD, SB-PD | Source PD/LGD models |
| Layer 2 | CECL-CRE, CECL-CNI, CECL-RES, CECL-CARDS, CECL-AUTO, CECL-SB | CECL reserve calculations |
| Layer 3 | ECON-CAPITAL | Economic capital aggregation |
| Layer 4 | STRESS-TEST, MODEL-MONITOR | Stress testing and monitoring |

```
Tool: map_model_dependencies
Input: model_id="CRE-PD-001", analysis_type="impact"

Output:
{
  "model_id": "CRE-PD-001",
  "downstream_dependencies": {
    "direct": {
      "count": 3,
      "models": ["CECL-CRE", "STRESS-TEST", "MODEL-MONITOR"],
      "impact": "Require revalidation if this model changes"
    },
    "indirect": {
      "count": 1,
      "models": ["ECON-CAPITAL"],
      "impact": "Require impact assessment if this model changes"
    }
  },
  "change_impact_assessment": {
    "models_requiring_revalidation": ["CECL-CRE", "STRESS-TEST", "MODEL-MONITOR"],
    "total_models_affected": 4,
    "percentage_of_portfolio": "25.0%",
    "estimated_effort": {
      "direct_revalidation": "240-360 hours",
      "impact_assessments": "20-40 hours"
    },
    "risk_rating": "HIGH"
  }
}
```

### SR 11-7 Compliance Checking

**Problem Solved:** "Does this model have complete documentation for the next regulatory exam?"

SR 11-7 is the Federal Reserve's guidance on Model Risk Management. The tool validates 9 required documentation elements:

1. Purpose and Scope
2. Data Sources
3. Methodology/Theoretical Basis
4. Variable Definitions (data dictionary)
5. Performance Metrics
6. Limitations and Assumptions
7. Governance and Approval
8. Ongoing Monitoring Plan
9. Change Log/Version History

```
Tool: check_sr11_compliance
Input: model_id="CRE-PD-001"

Output:
{
  "model_id": "CRE-PD-001",
  "sr11_compliance": {
    "overall_status": "SUBSTANTIALLY_COMPLIANT",
    "message": "Model has all required documentation but 1 element has incomplete sources",
    "score": "8/9 requirements fully met",
    "summary": {
      "compliant": 8,
      "partial": 1,
      "missing": 0
    }
  },
  "documentation_found": {
    "model_code": ["CRE_PD_v4.2.sas"],
    "data_dictionary": ["CRE_PD_data_dictionary.xlsx"],
    "white_paper": ["CRE_PD_Model_Documentation_v4.2.docx"],
    "meeting_minutes": ["Model_Validation_Committee_Minutes_Nov2024.docx"]
  },
  "remediation_plan": [
    {
      "priority": "MEDIUM",
      "requirement": "Performance Metrics",
      "action": "Complete documentation - missing: validation_report",
      "suggested_sources": ["validation_report"]
    }
  ]
}
```

### Presentation Consistency Checking

**Problem Solved:** "Is this committee presentation accurate, or does it show outdated thresholds?"

```
Tool: check_presentation_consistency
Input: pptx_filename="CRE_PD_Validation_Presentation.pptx", sas_filename="CRE_PD_v4.2.sas"

Output:
{
  "analysis": {
    "status": "NEEDS_UPDATE",
    "message": "Presentation has 2 inconsistencies that should be corrected",
    "total_slides": 12,
    "slides_with_issues": 2
  },
  "source_of_truth": {
    "dscr_threshold": 1.25,
    "ltv_threshold": 0.80,
    "segment_count": 4,
    "property_types": ["INDUSTRIAL", "MULTIFAMILY", "OFFICE", "RETAIL"]
  },
  "inconsistencies": [
    {
      "slide": 5,
      "type": "THRESHOLD_MISMATCH",
      "parameter": "DSCR",
      "presentation_value": 1.20,
      "code_value": 1.25,
      "severity": "HIGH",
      "recommendation": "Update DSCR threshold from 1.20 to 1.25"
    },
    {
      "slide": 3,
      "type": "SEGMENT_COUNT_MISMATCH",
      "parameter": "Property Segments",
      "presentation_value": 3,
      "code_value": 4,
      "severity": "MEDIUM",
      "recommendation": "Update segment count from 3 to 4"
    }
  ],
  "recommendation": "Update the slides listed above before presenting to Model Validation Committee"
}
```

---

## Prompt Templates

Beyond tools, the server provides rich **prompt templates** that gather context from multiple sources and structure it for complex analytical tasks:

### New Analyst Onboarding

```
Prompt: new_analyst_onboarding
Input: model_id="CRE-PD-001"
```

Generates a comprehensive onboarding brief by:
1. Calling `check_sr11_compliance` for compliance status
2. Calling `map_model_dependencies` for upstream/downstream context
3. Loading model registry metadata
4. Extracting SAS parameters
5. Listing available documentation

The LLM receives all this context and produces a structured onboarding document with:
- Executive Summary
- Key Things to Know on Day One
- Important Thresholds and Parameters
- Dependency Map
- Compliance Priorities
- Documentation Roadmap
- 30-60-90 Day Plan

### Model Change Impact Assessment

```
Prompt: model_change_impact_assessment
Input: model_name="CRE_PD", version_old="v3.5", version_new="v4.2"
```

Generates a comprehensive impact assessment by:
1. Comparing code between versions
2. Extracting performance sections from both version's documentation
3. Extracting variable documentation from both versions
4. Mapping downstream dependencies

The LLM produces an assessment with:
- Variable Changes Analysis
- Threshold and Segment Changes
- Performance Impact
- Loss Forecast Implications
- Downstream Impact
- Documentation Updates Required
- Risk Assessment

---

## Demo Scenarios

### Scenario 1: Pre-Committee Validation

**Context:** A model developer is presenting CRE-PD v4.2 to the Model Validation Committee tomorrow.

**Workflow:**
1. Run `check_presentation_consistency` to verify slides match current code
2. Run `check_sr11_compliance` to confirm documentation completeness
3. Run `compare_model_versions` to summarize changes from last approved version

**Result:** Developer discovers slide 5 shows outdated DSCR threshold (1.20 instead of 1.25), corrects it before committee, avoids embarrassing Q&A about inconsistencies.

### Scenario 2: Impact Assessment for Data Source Change

**Context:** The economics team is changing their unemployment rate forecast methodology. Which models are affected?

**Workflow:**
1. Run `map_model_dependencies` with `data_source="unemployment_rate"`
2. Review directly affected models (CRE-PD, STRESS-TEST)
3. Trace indirect impacts through dependency graph

**Result:** Team identifies 3 models requiring revalidation, 2 requiring impact assessment, estimates 280-400 hours of work, can plan resource allocation before implementing change.

### Scenario 3: New Analyst Onboarding

**Context:** A new model developer is joining the team and taking over responsibility for the CRE-PD model.

**Workflow:**
1. Run `new_analyst_onboarding` prompt template
2. LLM generates comprehensive briefing document
3. New analyst receives structured 30-60-90 day plan

**Result:** Instead of 2-3 weeks of tribal knowledge transfer, analyst has a structured document covering everything they need to know, with links to source documentation.

---

## Performance Results

### Tool Response Times

| Tool | Typical Response |
|------|------------------|
| `list_model_files` | < 100ms |
| `extract_sas_parameters` | 200-500ms |
| `read_excel_dictionary` | 300-700ms |
| `check_cross_file_consistency` | 800ms-1.5s |
| `compare_model_versions` | 1-2s |
| `extract_word_content` | 500ms-1s |
| `map_model_dependencies` | 200-500ms |
| `check_sr11_compliance` | 1-2s |
| `check_presentation_consistency` | 1-3s |

### Time Savings (Estimated)

| Task | Manual Time | With MCP Server |
|------|-------------|-----------------|
| Cross-file consistency check | 2-4 hours | 30 seconds |
| Version comparison | 4-8 hours | 2 minutes |
| SR 11-7 compliance check | 2-3 hours | 1 minute |
| Presentation validation | 1-2 hours | 30 seconds |
| Dependency impact analysis | 4-6 hours | 1 minute |
| New analyst onboarding brief | 2-3 days | 5 minutes |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Protocol** | MCP (Model Context Protocol) | Anthropic's standard for AI tool integration |
| **Language** | Python 3.10+ | Core development language |
| **Excel Parsing** | openpyxl | Data dictionary extraction |
| **Word Parsing** | python-docx | White paper and meeting minutes extraction |
| **PowerPoint Parsing** | python-pptx | Presentation content extraction |
| **SAS Parsing** | regex | Threshold and variable extraction |
| **Async Runtime** | asyncio | Non-blocking tool execution |
| **Testing** | pytest | Unit and integration tests |

---

## Project Structure

```
mcp_banking_workflows/
├── servers/
│   └── banking_filesystem_server.py    # MCP server (950+ lines, 10 tools)
├── mock_data/
│   ├── model_registry.json             # 16 models with full metadata
│   ├── models/                         # SAS and Python model code
│   │   ├── CRE_PD_v4.2.sas
│   │   ├── CRE_PD_v3.5.sas
│   │   └── CNI_PD_v5.0.py
│   ├── excel/                          # Data dictionaries
│   │   ├── CRE_PD_data_dictionary.xlsx
│   │   └── CNI_PD_data_dictionary.xlsx
│   ├── word/                           # White papers, meeting minutes
│   │   ├── CRE_PD_Model_Documentation_v4.2.docx
│   │   └── Model_Validation_Committee_Minutes_Nov2024.docx
│   └── presentations/                  # Committee presentations
│       └── CRE_PD_Validation_Presentation.pptx
├── tests/
│   ├── test_sr11_compliance.py
│   ├── test_dependency_mapping.py
│   ├── test_version_comparison_tool.py
│   ├── test_word_extraction.py
│   └── test_presentation_consistency.py
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/pmcavallo/mcp_banking_workflows.git
cd mcp_banking_workflows

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Direct execution
python servers/banking_filesystem_server.py

# With MCP Inspector (for testing)
# Windows
set CLIENT_PORT=6290 && set SERVER_PORT=6291 && npx @modelcontextprotocol/inspector py servers/banking_filesystem_server.py
```

### Claude Desktop Integration

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "banking-mrm": {
      "command": "python",
      "args": ["C:/Projects/mcp_banking_workflows/servers/banking_filesystem_server.py"]
    }
  }
}
```

---

## What This Demonstrates

### MCP Implementation Patterns

* **Tool Design:** 10 tools with clear input/output contracts
* **Multi-Source Integration:** Unified access to SAS, Excel, Word, PowerPoint
* **Structured Output:** JSON responses that LLMs can reason about
* **Prompt Templates:** Rich context gathering for complex analytical tasks

### Model Risk Management Expertise

* **SR 11-7 Compliance:** Understanding of Federal Reserve documentation requirements
* **Dependency Analysis:** Graph-based impact assessment for interconnected model portfolios
* **Version Control:** Systematic comparison of model iterations
* **Cross-Format Validation:** Consistency checking across code and documentation

### Production AI Patterns

* **Async Architecture:** Non-blocking tool execution
* **Error Handling:** Graceful degradation with informative messages
* **File Format Agnostic:** Native parsing of banking-standard formats
* **Extensibility:** Clear patterns for adding new tools and prompts

---

## Future Enhancements

**Phase 1: Enhanced Parsing**
- Support for R and Python model code (beyond SAS)
- Multi-sheet Excel workbook handling
- PDF extraction for regulatory filings

**Phase 2: Write Operations**
- Generate data dictionary from SAS code
- Auto-update PowerPoint slides with current thresholds
- Create SR 11-7 gap remediation templates

**Phase 3: Integration**
- Connect to model risk management platforms (SAS Model Manager, Moody's RiskAuthority)
- Integration with Jira/ServiceNow for issue tracking
- Automated monitoring alerts for documentation drift

---

## License

MIT License - see LICENSE file for details.

---

*"The difference between good and great model risk management isn't the accuracy of the models - it's the accuracy of the documentation. In regulated banking, a single inconsistency between code and committee presentation can delay an approval by months. MCP Banking Workflows solves this by giving AI assistants the tools to validate what humans miss."*

