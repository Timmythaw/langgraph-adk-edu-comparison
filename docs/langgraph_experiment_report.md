# AI Teacher Assistant — LangGraph Implementation Report

> This document describes the actual implementation of the LangGraph experiment
> as executed in `notebooks/01_langgraph_system.ipynb`. It serves as the
> technical reference for the empirical comparison research at Mae Fah Luang University.

---

## 1. Overview

This notebook implements an AI Teaching Assistant using **LangGraph v1.1.2**,
running on **Google Colab** with **Vertex AI** as the backend. It is one of two
parallel implementations (LangGraph vs Google ADK) used for empirical comparison
in the research paper.

### Research Context
- **Paper:** Comparative analysis of LangGraph vs Google ADK for Educational Context
- **Institution:** Mae Fah Luang University (MFU), Thailand
- **Evaluation:** 3 test scenarios, 5 runs each (15 total) for latency; 3 runs per scenario for quality evaluation
- **Goal:** Provide empirical findings for the research paper

### System Purpose
Automate three instructional tasks for university lecturers:
1. **Lesson Plan Generation** — grounded in course materials via Vertex AI Search
2. **Quiz Generation** — 10-question MCQ, output formatted for instructor review
3. **Student Email Drafting** — HITL-gated before sending, with optional in-place editing

---

## 2. Environment & Dependencies

| Component | Value |
|---|---|
| Runtime | Google Colab (Python 3.12) |
| LangGraph Version | `langgraph 1.1.2` |
| LLM — Orchestrator | `gemini-2.5-pro` (via `ChatGoogleGenerativeAI`, `vertexai=True`) |
| LLM — Worker nodes | `gemini-2.5-flash` (via `ChatGoogleGenerativeAI`, `vertexai=True`) |
| RAG Backend | Vertex AI Search (Discovery Engine) |
| Google Cloud Project | (Configured via Colab Secrets) |
| Search Location | `global` |
| Datastore | (GCS-backed, configured via Colab Secrets) |
| Vertex AI Init Location | `us-central1` |
| Auth | `google.colab.auth.authenticate_user()` |
| Checkpointer | `MemorySaver` (in-memory) |
| Tracing | LangSmith (`LANGCHAIN_TRACING_V2=true`) |
| LangSmith Project | `langgraph-adk-edu-comparison` |
| `max_output_tokens` | `4096` (both models) |

> **Note:** `ChatGoogleGenerativeAI` with `vertexai=True` is the correct current
> import from `langchain-google-genai`. The old report referenced `ChatVertexAI`
> which is deprecated — the notebook uses `ChatGoogleGenerativeAI` throughout.

---

## 3. Agent Architecture

The system uses LangGraph's **StateGraph** with a typed `TeacherState`, a
keyword+LLM hybrid router node, and conditional edges routing to specialist
nodes. Control flow is **deterministic** — fully defined by graph edges,
not LLM discretion.

```
START
└── router (gemini-2.5-pro, keyword-first then LLM fallback)
    ├── lessonplanner ────────────────────────────────────► END
    ├── quizcontent ──► quizpublisher ──────────────────► END
    └── emaildrafter ──► hitlapproval
                          └── emailsender ──────────► END
```

> **Note:** Actual node names in the compiled graph are `lessonplanner`,
> `quizcontent`, `quizpublisher`, `emaildrafter`, `hitlapproval`, `emailsender`
> (no underscores) — as confirmed by `graph.get_graph().nodes.keys()` output.

### State Schema (`TeacherState`)

| Field | Type | Description |
|---|---|---|
| `messages` | `Annotated[list, operator.add]` | Append-only conversation history |
| `task_type` | `str` | `lessonplan`, `quiz`, or `email` |
| `course_materials` | `str` | RAG retrieval result |
| `draft_output` | `str` | Intermediate draft (email or quiz JSON) |
| `final_output` | `str` | Final formatted response |
| `hitl_decision` | `str` | `"approved"` or `"rejected"` |

> **Note:** `task_type` values are `lessonplan` (no underscore), `quiz`, and
> `email` — matching the keyword mapping in `router_node`.

### Key LangGraph Advantages Observed

| Feature | Impact |
|---|---|
| Explicit typed state | All nodes share and mutate a single `TeacherState` dict — no `output_key` workarounds needed |
| Deterministic routing | `add_conditional_edges` makes control flow explicit and auditable |
| HITL with edit support | Instructor can approve, reject, **or paste an edited draft** — 3-way decision vs ADK's binary |
| No `output_schema` + `tools` constraint | Quiz pipeline implemented cleanly without architectural workarounds |
| `MemorySaver` checkpointer | Full graph state persisted across interrupts |

---

## 4. RAG Tool — Vertex AI Search

`retrieve_course_materials(query, page_size, summary_result_count)` queries
the Vertex AI Search datastore using `discoveryengine.SearchServiceClient`.
Accepts dynamic `page_size` and `summary_result_count` parameters — each node
calls it with different values depending on task depth:

| Node | `page_size` | `summary_result_count` |
|---|---|---|
| `lesson_planner_node` | 6 | 5 |
| `quiz_content_node` | 8 | 5 |
| `email_drafter_node` | 2 | 2 |

Returns snippets joined by `\n\n---\n\n`, or `"No relevant materials found."`.
All nodes are decorated with `@traceable` for individual LangSmith node tracing.

Verified during setup: successfully retrieved course material snippets for
`"software testing Textbook."`.

---

## 5. Node Details

All nodes are decorated with `@traceable(run_type="chain", metadata={"framework": "LangGraph", "node": "..."})`.

### Router Node (`router`)
- **Model:** `orchestrator_llm` (`gemini-2.5-pro`), used only in LLM fallback stage
- **Strategy:** Two-stage routing
  1. **Fast keyword match** (checked in this order — email has highest priority):
     - `email/send/announcement/draft` → `email`
     - `lesson plan/outline/lecture` → `lessonplan`
     - `quiz/multiple choice/question/mcq` → `quiz`
  2. **LLM fallback** — if no keywords match, `gemini-2.5-pro` classifies intent; raw response validated against `["lessonplan", "quiz", "email"]`; defaults to `lessonplan` if invalid
- **Routing function:** `route_to_agent(state)` returns `Literal["lessonplanner", "quizcontent", "emaildrafter"]`

### Lesson Planner Node (`lessonplanner`)
- **Model:** `worker_llm` (`gemini-2.5-flash`)
- **RAG:** `page_size=6, summary_result_count=5`
- **Output:** Full 90-minute lesson plan written to `final_output` and appended to `messages`

### Quiz Content Node (`quizcontent`)
- **Model:** `worker_llm` (`gemini-2.5-flash`)
- **RAG:** `page_size=8, summary_result_count=5`
- **Output:** 10 MCQs as a JSON array written to `draft_output`

### Quiz Publisher Node (`quizpublisher`)
- **Model:** `worker_llm` (`gemini-2.5-flash`)
- **Input:** `draft_output` (quiz JSON from state)
- **Output:** Formatted numbered list with ✓ correct answers written to `final_output` and appended to `messages`

### Email Drafter Node (`emaildrafter`)
- **Model:** `worker_llm` (`gemini-2.5-flash`)
- **RAG:** `page_size=2, summary_result_count=2`
- **Output:** Formal email (`SUBJECT:` / `BODY:` format) written to `draft_output`

### HITL Approval Node (`hitlapproval`)
- **Mechanism:** Python `input()` pauses graph execution
- **Decision options:**
  - `yes` → `hitl_decision = "approved"`
  - `no` → `hitl_decision = "rejected"`
  - any other text → treated as edited draft, replaces `draft_output`, `hitl_decision = "approved"`
- **Key difference vs ADK:** Instructors can edit the draft in-place — this path was architecturally available but not exercised during the experiment runs
- **Routing after HITL:** `route_after_hitl()` always returns `"emailsender"` — sender handles both approved/rejected internally

### Email Sender Node (`emailsender`)
- **Model:** None (pure state logic)
- **Behavior:** Checks `hitl_decision`; writes confirmation or rejection notice to `final_output` and appends to `messages`

---

## 6. Graph Assembly

```python
checkpointer = MemorySaver()
builder = StateGraph(TeacherState)

builder.add_node("router",        router_node)
builder.add_node("lessonplanner", lesson_planner_node)
builder.add_node("quizcontent",   quiz_content_node)
builder.add_node("quizpublisher", quiz_publisher_node)
builder.add_node("emaildrafter",  email_drafter_node)
builder.add_node("hitlapproval",  hitl_approval_node)
builder.add_node("emailsender",   email_sender_node)

builder.set_entry_point("router")

builder.add_conditional_edges("router", route_to_agent)
builder.add_edge("lessonplanner", END)
builder.add_edge("quizcontent",   "quizpublisher")
builder.add_edge("quizpublisher", END)
builder.add_edge("emaildrafter",  "hitlapproval")
builder.add_conditional_edges("hitlapproval", route_after_hitl)
builder.add_edge("emailsender",   END)

graph = builder.compile(checkpointer=checkpointer)
```

Compiled nodes: `__start__`, `router`, `lessonplanner`, `quizcontent`,
`quizpublisher`, `emaildrafter`, `hitlapproval`, `emailsender`, `__end__`

---

## 7. Runner Setup

```python
APP_NAME = "teacher-assistant-langgraph"
USERID   = "mfu-instructor-01"
```

`run_request(user_input, scenario)` is decorated with `@traceable` for LangSmith
tracing. It creates a fresh UUID `thread_id` per run, passes scenario metadata
in `config`, invokes the graph synchronously via `graph.invoke()`, and captures
the LangSmith run ID from `get_current_run_tree()`.

**Token counts:** LangGraph does not expose per-message token counts natively via `graph.invoke()`.
`input_tokens` and `output_tokens` are set to `-1` as a sentinel value — unlike
the ADK notebook which captures `usage_metadata` from events. Token data is
accessible via LangSmith traces.

Latency is sourced preferentially from LangSmith via `fetch_ls_latency(ls_run_id)`,
which polls with `retries=6, delay=5.0s` before falling back to `time.time()`
wall-clock.

---

## 8. Metrics Logger

Per-run data recorded to `langgraph_metrics.csv`:

| Field | Description |
|---|---|
| `timestamp` | UTC ISO timestamp |
| `scenario` | Scenario name |
| `framework` | Always `"LangGraph"` |
| `routing_correct` | Boolean — correct node invoked? |
| `latency_sec` | LangSmith latency or `time.time()` fallback |
| `latency_source` | `"langsmith"` or `"fallback"` |
| `input_tokens` | Always `-1` (not natively available in LangGraph) |
| `output_tokens` | Always `-1` (not natively available in LangGraph) |
| `response_length` | Character count of `final_output` |
| `error` | Error message if run failed |

---

## 9. Experiment Scenarios

### Scenario 1 — Lesson Plan Generation
> *"Create a 90-minute lesson plan on Software Testing for second-year Software Engineering students. Align it with the course materials."*
- 5 runs | Sleep: 30s between runs | Routing check: `"lesson plan"` or `"learning objectives"` in response

### Scenario 2 — Quiz Generation
> *"Generate 10 multiple-choice questions on Software Testing from the course materials."*
- 5 runs | Sleep: 30s between runs | Routing check: `"question"` or `"quiz"` in response

### Scenario 3 — Email with HITL
> *"Draft and send an email to all students reminding them that the SQL Joins quiz is next Monday at 9am. Include what topics to study."*
- 5 runs | Real instructor `input()` per run | HITL decisions: `yes, no, yes, yes, yes`
- Routing check: `"email"`, `"approved"`, `"rejected"`, or `"not sent"` in response
- Latency is LangSmith end-to-end measurement (includes human review + typing time)

---

## 10. Results Summary

All 15 runs completed successfully with **100% routing accuracy**. All latency
values are LangSmith-sourced (100% `langsmith` source rate across all scenarios).

| Scenario | Runs | Avg Latency | Min | Max | Routing |
|---|---|---|---|---|---|
| Scenario 1 — Lesson Plan | 5 | 26.17s | 17.18s | 32.00s | 100% |
| Scenario 2 — Quiz Generation | 5 | 25.50s | 16.89s | 28.17s | 100% |
| Scenario 3 — Email HITL | 5 | 10.74s | 9.44s | 12.73s | 100% |
| **Overall** | **15** | **20.80s** | — | — | **100%** |

### Per-Run Detail

| Run | Scenario | Latency (s) | Source |
|---|---|---|---|
| 1 | Lesson Plan | 27.78 | langsmith |
| 2 | Lesson Plan | 17.18 | langsmith |
| 3 | Lesson Plan | 26.93 | langsmith |
| 4 | Lesson Plan | 26.97 | langsmith |
| 5 | Lesson Plan | 32.00 | langsmith |
| 6 | Quiz Generation | 28.12 | langsmith |
| 7 | Quiz Generation | 27.55 | langsmith |
| 8 | Quiz Generation | 26.76 | langsmith |
| 9 | Quiz Generation | 16.89 | langsmith |
| 10 | Quiz Generation | 28.17 | langsmith |
| 11 | Email HITL | 12.73 | langsmith |
| 12 | Email HITL | 9.44 | langsmith |
| 13 | Email HITL | 10.13 | langsmith |
| 14 | Email HITL | 10.53 | langsmith |
| 15 | Email HITL | 10.87 | langsmith |

### Key Observations
- **Lesson Plan produced the longest responses** — significantly more verbose than ADK
- **Email HITL was fastest** (10.74s avg) — light RAG (`page_size=2`) and minimal LLM output; HITL decision time included
- **Scenario 3 latency variance is low** (9.44–12.73s) — HITL decisions were answered promptly; reflect real human review time
- **Two-stage routing worked perfectly** — all 15 runs routed correctly on first keyword match; LLM fallback was never triggered
- **HITL edit path available but not exercised** — all decisions were binary `yes`/`no`; the edit path is architecturally supported
- **Token counts unavailable natively** — all `input_tokens` and `output_tokens` logged as `-1`; token data accessible via LangSmith traces only
- **All latency values are LangSmith-sourced** — 100% clean latency, no fallback measurements recorded

---

## 11. Quality Evaluation Results (LLM-as-a-Judge — Notebook 03)

3 runs per scenario evaluated by `gemini-2.5-pro` on 5 criteria (max 5 each, total 25).

| Scenario | Runs | Avg Total | Accuracy | Completeness | Clarity | Edu Relevance | Task Adherence |
|---|---|---|---|---|---|---|---|
| Lesson Plan | 3 | 25.00 | 5.0 | 5.00 | 5.0 | 5.00 | 5.00 |
| Quiz Generation | 3 | 22.33 | 5.0 | 5.00 | 5.0 | 2.33 | 5.00 |
| Email HITL | 3 | 25.00 | 5.0 | 5.00 | 5.0 | 5.00 | 5.00 |
| **Overall** | **9** | **24.11** | **5.0** | **5.00** | **5.00** | **4.11** | **5.00** |

> Quiz edu_relevance (2.33) was lower due to repetitive questions derived from a small set of retrieved chunks.
> LangGraph achieved perfect scores in completeness, clarity, and task_adherence overall.

---

## 12. RAG Faithfulness (RAGAS — Notebook 04)

| Scenario | Faithfulness | Answer Relevancy | Context Precision | Contexts Retrieved |
|---|---|---|---|---|
| Lesson Plan | 0.964 | 0.797 | 0.000 | 2 chunks |
| Quiz | 0.115 | 0.647 | 0.000 | 2 chunks |
| Email | 0.110 | 0.581 | 0.000 | 2 chunks |

> Lesson plan faithfulness (0.964) is the highest across both frameworks, indicating strong grounding in retrieved course materials.

---

## 13. Hallucination Analysis (Notebook 05)

| Scenario | Groundedness | RAGAS Faithfulness | Composite Halluc % |
|---|---|---|---|
| Lesson Plan | 1.0 | 0.964 | **34.5%** |
| Quiz | 1.0 | 0.115 | 62.8% |
| Email | 1.0 | 0.110 | 63.0% |

> LangGraph achieves the lowest composite hallucination score in lesson plan generation (34.5%) across both frameworks.

---

## 14. Output Files

| File | Description |
|---|---|
| `langgraph_metrics.csv` | Raw per-run latency metrics (15 rows) |
| `langgraph_latency_chart.png` | Bar chart of avg latency by scenario with LangSmith source % |
| `judge_results.csv` | Per-run LLM Judge scores |
| `judge_radar_overall.png` | Radar chart comparing LangGraph vs ADK quality dimensions |
| `ragas_comparison_chart.png` | RAGAS metric comparison chart |
| `hallucination_results.csv` | Per-run hallucination scores |
| `hallucination_composite.csv` | Aggregated composite hallucination by framework and scenario |
| `hallucination_heatmap.png` | Heatmap of composite hallucination % |
| `01_langgraph_system.ipynb` | Full experiment notebook with cell outputs |
