# LangGraph vs Google ADK — AI Teacher Assistant

> **Implementation and evaluation code for:**
> *"A Comparison Framework for LangGraph and Google ADK: Agentic AI Implementation in Educational Context"*
> — Mae Fah Luang University, Thailand (2026)

[![Open In Colab — LangGraph](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Timmythaw/langgraph-adk-edu-comparison/blob/main/notebooks/01_langgraph_system.ipynb)
[![Open In Colab — ADK](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Timmythaw/langgraph-adk-edu-comparison/blob/main/notebooks/02_adk_system.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains two **parallel, functionally-equivalent** AI Teaching
Assistant implementations — one built with **LangGraph 1.1.2** and one with
**Google ADK 1.27.1** — evaluated against identical test scenarios. Both systems
assist university lecturers at Mae Fah Luang University with three core tasks:

| Task | Description |
|---|---|
| **Lesson Plan Generation** | Grounded 90-minute lesson plans from Vertex AI Search course materials |
| **Quiz Generation** | 10-question MCQ assessments, formatted for instructor review |
| **Student Email with HITL** | Draft → Human-in-the-Loop approval → Send confirmation |

Both use **Gemini 2.5 Pro** as the orchestrator and **Gemini 2.5 Flash** as
worker agents, backed by **Vertex AI Search** for RAG over course materials.
All runs are traced end-to-end via **LangSmith** (project: `langgraph-adk-edu-comparison`).

---

## Architecture

### LangGraph — Stateful Graph with Explicit Routing
```
└── router (keyword-first, LLM fallback via gemini-2.5-pro)
├── lessonplanner ────────────────────────────────────► END
├── quizcontent ──► quizpublisher ───────────────────► END
└── emaildrafter ──► hitlapproval
└── emailsender ────────────► END
```

- **Control flow:** Deterministic via `add_conditional_edges`
- **State:** Typed `TeacherState` dict — shared across all nodes (`messages`, `task_type`, `course_materials`, `draft_output`, `final_output`, `hitl_decision`)
- **HITL:** `input()` pause — instructor can **approve (`yes`), reject (`no`), or paste an edited draft**
- **Checkpointing:** `MemorySaver` persists full state across interrupts
- **Token tracking:** Not natively available via `graph.invoke()` — tokens logged as `-1`; available via LangSmith traces

### Google ADK — Agent-as-Tool with AutoFlow
```
RootOrchestrator (LlmAgent, gemini-2.5-pro)
│ ← AutoFlow routing
├── lesson_planner_agent (LlmAgent, gemini-2.5-flash)
│ └── retrieve_course_materials (Vertex AI Search)
├── quiz_generator_agent (SequentialAgent)
│ ├── quiz_content_agent → output_key="quiz_questions_json"
│ └── quiz_publisher_agent → reads {quiz_questions_json}
└── email_agent (SequentialAgent)
├── email_drafter_agent → output_key="email_draft"
└── email_sender_agent → before_tool_callback HITL + send_email_to_students
```

- **Control flow:** Non-deterministic AutoFlow — Gemini reads agent `description` fields
- **State:** Passed via `output_key` between sequential agents; `InMemorySessionService`
- **HITL:** Binary approve/reject only via `before_tool_callback` — no in-place draft editing
- **Resilience:** `HttpRetryOptions` (5 attempts, 5s initial delay, covers 408/429/5xx)
- **Note:** `output_schema` + `tools` cannot be combined in a single ADK `LlmAgent` — Quiz generation requires a 2-agent `SequentialAgent` workaround

---

## Results

All **30 runs** (5 per scenario × 3 scenarios × 2 frameworks) completed with
**100% routing accuracy**. All latency values are **LangSmith-sourced** (100%
clean latency, no `time.time()` fallback measurements recorded in either notebook).

### Latency by Scenario

| Scenario | LangGraph Avg | ADK Avg | Winner |
|---|---|---|---|
| Lesson Plan | 26.17s | 24.82s | ADK ✓ |
| Quiz Generation | 25.50s | 26.39s | LangGraph ✓ |
| Email HITL | 10.74s | 14.93s | LangGraph ✓ |
| **Overall** | **20.80s** | **22.05s** | **LangGraph** ✓ |

> All latency values are LangSmith end-to-end measurements. Scenario 3 includes
> real human review + typing time for HITL decisions.

### LangGraph Per-Run Detail

| Run | Scenario | Latency (s) |
|---|---|---|
| 1 | Lesson Plan | 27.78 |
| 2 | Lesson Plan | 17.18 |
| 3 | Lesson Plan | 26.93 |
| 4 | Lesson Plan | 26.97 |
| 5 | Lesson Plan | 32.00 |
| 6 | Quiz Generation | 28.12 |
| 7 | Quiz Generation | 27.55 |
| 8 | Quiz Generation | 26.76 |
| 9 | Quiz Generation | 16.89 |
| 10 | Quiz Generation | 28.17 |
| 11 | Email HITL | 12.73 |
| 12 | Email HITL | 9.44 |
| 13 | Email HITL | 10.13 |
| 14 | Email HITL | 10.53 |
| 15 | Email HITL | 10.87 |

### ADK Per-Run Detail

| Run | Scenario | Latency (s) |
|---|---|---|
| 1 | Lesson Plan | 18.44 |
| 2 | Lesson Plan | 19.85 |
| 3 | Lesson Plan | 26.39 |
| 4 | Lesson Plan | 31.97 |
| 5 | Lesson Plan | 27.44 |
| 6 | Quiz Generation | 24.13 | 
| 7 | Quiz Generation | 32.27 |
| 8 | Quiz Generation | 19.21 | 
| 9 | Quiz Generation | 28.69 |
| 10 | Quiz Generation | 27.63 |
| 11 | Email HITL | 13.97 |
| 12 | Email HITL | 15.32 |
| 13 | Email HITL | 15.15 |
| 14 | Email HITL | 12.49 |
| 15 | Email HITL | 17.70 |

### Latency Charts

| LangGraph | Google ADK |
|---|---|
| ![LangGraph Latency](output/langgraph_latency.png) | ![ADK Latency](output/adk_latency.png) |

### Side-by-Side Comparison

![LangGraph vs ADK](output/LG%20VS%20ADK.png)

### Key Findings

| Dimension | LangGraph | Google ADK |
|---|---|---|
| Routing | Deterministic (keyword-first, LLM fallback) | Non-deterministic (AutoFlow) |
| HITL flexibility | Approve / Reject / **Edit in-place** | Approve / Reject only (binary) |
| State management | Explicit typed `TeacherState` shared across all nodes | `output_key` chaining between sequential agents |
| Framework constraint | None observed | `output_schema` + `tools` conflict → 2-agent `SequentialAgent` workaround required |
| Token tracking | Not available natively (`-1` sentinel) | Available via `usage_metadata` events |
| Latency resilience | No retry logic | `HttpRetryOptions` (5 attempts, covers 429/5xx) |
| Response verbosity | Higher (lesson plans significantly longer) | Lower/more concise |
| Overall avg latency | **20.80s** | 22.05s |

---

## Repository Structure
```
langgraph-adk-edu-comparison/
├── notebooks/
│ ├── 01_langgraph_system.ipynb # LangGraph implementation + experiment
│ └── 02_adk_system.ipynb # Google ADK implementation + experiment
├── docs/
│ ├── langgraph_experiment_report.md # Detailed technical report — LangGraph
│ └── adk_experiment_report.md # Detailed technical report — ADK
├── output/
│ ├── langgraph_metrics.csv # Raw per-run data (15 rows)
│ ├── adk_metrics.csv # Raw per-run data (15 rows)
│ ├── langgraph_latency_chart.png # Latency bar chart — LangGraph
│ ├── adk_latency_chart.png # Latency bar chart — ADK
│ └── LG VS ADK.png # Side-by-side comparison chart
├── .env.example # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud project with **Vertex AI API** and **Discovery Engine API** enabled
- A Vertex AI Search datastore populated with course materials
- LangSmith account for tracing (used by both notebooks — project: `langgraph-adk-edu-comparison`)

### 1. Clone & Configure

```bash
git clone https://github.com/Timmythaw/langgraph-adk-edu-comparison.git
cd langgraph-adk-edu-comparison
cp .env.example .env
# Edit .env with your GCP project ID, datastore ID, and credentials
```

### 2. Environment Variables

```bash
# Shared
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global                    # Vertex AI Search location
VERTEX_AI_SEARCH_DATASTORE_ID=your-datastore-id
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

# LangSmith tracing (used by both notebooks)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=langgraph-adk-edu-comparison
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# ADK only
GOOGLE_GENAI_USE_VERTEXAI=1
LANGSMITH_API_KEY=your-langsmith-api-key        # ADK uses langsmith package directly
```

### 3. Run in Google Colab (Recommended)

The notebooks are self-contained and designed for Colab. Click the badges at the top of this README, then:

1. Store `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `VERTEX_AI_SEARCH_DATASTORE_ID`, and `LANGSMITH_API_KEY` as **Colab Secrets**
2. Run all cells in order
3. During Scenario 3 (Email HITL):
   - LangGraph: type `yes` to approve, `no` to reject, or **paste edited text** to approve with changes
   - ADK: type `yes` to approve or anything else to reject

---

## Documentation

- [LangGraph Experiment Report](docs/langgraph_experiment_report.md) — implementation details, node descriptions, and full results
- [ADK Experiment Report](docs/adk_experiment_report.md) — implementation details, agent architecture, and full results

---

## Tech Stack

| Component | LangGraph | Google ADK |
|---|---|---|
| Framework | `langgraph 1.1.2` | `google-adk 1.27.1` |
| Orchestrator LLM | `gemini-2.5-pro` (`ChatGoogleGenerativeAI`, `vertexai=True`) | `gemini-2.5-pro` (`Gemini`, `resilient_pro`) |
| Worker LLM | `gemini-2.5-flash` (`ChatGoogleGenerativeAI`, `vertexai=True`) | `gemini-2.5-flash` (`Gemini`, `resilient_flash`) |
| RAG | Vertex AI Search (Discovery Engine) | Vertex AI Search (Discovery Engine) |
| State | `MemorySaver` + typed `TeacherState` | `InMemorySessionService` + `output_key` chaining |
| Retry logic | None | `HttpRetryOptions` (5 attempts, 429/5xx) |
| Tracing | `LANGCHAIN_TRACING_V2` (native) | `langsmith.integrations.google_adk.configure_google_adk()` |
| Runtime | Google Colab / Python 3.12 | Google Colab / Python 3.12 |

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{thawzinmyoaung_thanthtoosan2026lgadk,
  title  = {A Comparison Framework for LangGraph and Google ADK: Agentic AI Implementation in Educational Context},
  author = {Thaw Zin Myo Aung and Thant Htoo San},
  year   = {2026},
  url    = {https://github.com/Timmythaw/langgraph-adk-edu-comparison}
}
```

---

## License

[MIT License](LICENSE) — Mae Fah Luang University, Thailand, 2026