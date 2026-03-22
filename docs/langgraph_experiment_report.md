# AI Teacher Assistant — LangGraph Implementation Report

> This document describes the actual implementation of the LangGraph experiment as executed in `notebooks/01_langgraph_system.ipynb`. It serves as the technical reference for the empirical comparison research at Mae Fah Luang University.

---

## 1. Overview

This notebook implements an AI Teaching Assistant using **LangGraph v1.1.2**, running on **Google Colab** with **Vertex AI** as the backend. It is one of two parallel implementations (LangGraph vs Google ADK) used for empirical comparison in the research paper.

### Research Context
- **Paper:** Comparative analysis of LangGraph vs Google ADK for Educational Context
- **Institution:** Mae Fah Luang University (MFU), Thailand
- **Evaluation:** 3 test scenarios, 5 runs each (15 total)
- **Goal:** Provide empirical findings for Section V of the research paper

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
| LLM — Orchestrator | `gemini-2.5-pro` (via `ChatVertexAI`) |
| LLM — Worker nodes | `gemini-2.5-flash` (via `ChatVertexAI`) |
| RAG Backend | Vertex AI Search (Discovery Engine) |
| Google Cloud Project | `edu-teacher-assistant-prod` |
| Search Location | `global` |
| Datastore | `curriculum connector at GCS` |
| Auth | `google.colab.auth.authenticate_user()` |
| Checkpointer | `MemorySaver` (in-memory) |

> **Note:** `ChatVertexAI` is deprecated as of LangChain 3.2.0. A `DeprecationWarning` is shown at runtime but does not affect execution. Migration to `ChatGoogleGenerativeAI` is recommended for future work.

---

## 3. Agent Architecture

The system uses LangGraph's **StateGraph** with a typed `TeacherState`, a keyword+LLM hybrid router node, and conditional edges routing to specialist nodes. Control flow is **deterministic** — fully defined by graph edges, not LLM discretion.

```
START
  └── router_node (gemini-2.5-pro, keyword-first then LLM fallback)
        ├── lesson_planner_node  ──────────────────────────────► END
        ├── quiz_content_node ──► quiz_publisher_node ──────────► END
        └── email_drafter_node ──► hitl_approval_node
                                        └── email_sender_node ──► END
```

### State Schema (`TeacherState`)

| Field | Type | Description |
|---|---|---|
| `messages` | `list` (append-only) | Full conversation history |
| `task_type` | `str` | `lesson_plan`, `quiz`, or `email` |
| `course_materials` | `str` | RAG retrieval result |
| `draft_output` | `str` | Intermediate draft (email or quiz JSON) |
| `final_output` | `str` | Final formatted response |
| `hitl_decision` | `str` | `approved` or `rejected` |

### Key LangGraph Advantages Observed

| Feature | Impact |
|---|---|
| Explicit typed state | All nodes share and mutate a single `TeacherState` dict — no `output_key` workarounds needed |
| Deterministic routing | `add_conditional_edges` makes control flow explicit and auditable |
| HITL with edit support | Instructor can approve, reject, **or paste an edited draft** — 3-way decision vs ADK's binary |
| No `output_schema` + `tools` constraint | Quiz generation implemented as a single 2-node pipeline without workarounds |
| `MemorySaver` checkpointer | Full graph state persisted across interrupts |

---

## 4. RAG Tool — Vertex AI Search

`retrieve_course_materials(query)` queries the Vertex AI Search datastore, fetches the top 3 relevant snippets, and returns them as a concatenated string. Verified during setup: successfully retrieved course material snippets for `"software testing lesson plan"`.

---

## 5. Node Details

### Router Node
- **Models:** `gemini-2.5-pro` (LLM fallback only)
- **Strategy:** Two-stage routing
  1. **Fast keyword match** — `email/send/announcement/draft` → `email`; `lesson plan/outline/lecture` → `lesson_plan`; `quiz/multiple choice/question/mcq` → `quiz`
  2. **LLM fallback** — if no keyword matches, `gemini-2.5-pro` classifies the intent
- **Routing function:** `route_to_agent()` returns `Literal["lesson_planner", "quiz_content", "email_drafter"]`

### Lesson Planner Node
- **Model:** `gemini-2.5-flash`
- **Tools:** `retrieve_course_materials`
- **Behavior:** Fetches course content via RAG, generates a structured 90-minute lesson plan with learning objectives, timing, teaching methods, assessment strategy, and materials.

### Quiz Content Node
- **Model:** `gemini-2.5-flash`
- **Tools:** `retrieve_course_materials`
- **Output:** 10 MCQs as a JSON array stored in `draft_output`

### Quiz Publisher Node
- **Model:** `gemini-2.5-flash`
- **Input:** `draft_output` (quiz JSON from state)
- **Output:** Formatted numbered list with ✅ correct answers stored in `final_output`

### Email Drafter Node
- **Model:** `gemini-2.5-flash`
- **Output:** Formal email with `SUBJECT` and `BODY` stored in `draft_output`

### HITL Approval Node
- **Mechanism:** Python `input()` pauses graph execution
- **Decision options:**
  - `yes` → `hitl_decision = "approved"`
  - `no` → `hitl_decision = "rejected"`
  - any other text → treated as edited draft, replaces `draft_output`, `hitl_decision = "approved"`
- **Key difference vs ADK:** Instructors can **edit the draft in-place** by pasting modified text, not just approve or reject

### Email Sender Node
- **Behavior:** Checks `hitl_decision`; responds with confirmation (`approved`) or rejection notice (`rejected`) stored in `final_output`

---

## 6. Graph Assembly

```python
builder = StateGraph(TeacherState)

# Nodes
builder.add_node("router", router_node)
builder.add_node("lesson_planner", lesson_planner_node)
builder.add_node("quiz_content", quiz_content_node)
builder.add_node("quiz_publisher", quiz_publisher_node)
builder.add_node("email_drafter", email_drafter_node)
builder.add_node("hitl_approval", hitl_approval_node)
builder.add_node("email_sender", email_sender_node)

# Entry
builder.set_entry_point("router")

# Edges
builder.add_conditional_edges("router", route_to_agent)
builder.add_edge("lesson_planner", END)
builder.add_edge("quiz_content", "quiz_publisher")
builder.add_edge("quiz_publisher", END)
builder.add_edge("email_drafter", "hitl_approval")
builder.add_conditional_edges("hitl_approval", route_after_hitl)
builder.add_edge("email_sender", END)

graph = builder.compile(checkpointer=checkpointer)
```

Compiled nodes: `__start__`, `router`, `lesson_planner`, `quiz_content`, `quiz_publisher`, `email_drafter`, `hitl_approval`, `email_sender`, `__end__`

---

## 7. Runner Setup

A `MemorySaver` checkpointer and `run_request()` helper initialised with `app_name="teacher_assistant_langgraph"` and `user_id="mfu_instructor_01"`. Each run creates a fresh UUID `thread_id` and invokes the graph synchronously via `graph.invoke()`, returning `(final_output, latency)`.

---

## 8. Metrics Logger

Per-run data recorded to `langgraph_metrics.csv`:

| Field | Description |
|---|---|
| `timestamp` | UTC ISO timestamp |
| `scenario` | Scenario name |
| `framework` | Always `"LangGraph"` |
| `routing_correct` | Boolean — correct node invoked? |
| `latency_sec` | Wall-clock time to final response (includes HITL human input time) |
| `response_length` | Character count of response |
| `error` | Error message if run failed |

---

## 9. Experiment Scenarios

### Scenario 1 — Lesson Plan Generation
> *"Create a 90-minute lesson plan on Software Testing for second-year Software Engineering students. Align it with the course materials."*
- 5 runs | Routing check: `"lesson plan"` or `"learning objectives"` in response

### Scenario 2 — Quiz Generation
> *"Generate 10 multiple-choice questions on Software Testing from the course materials."*
- 5 runs | Routing check: `"question"` or `"quiz"` in response

### Scenario 3 — Email with HITL
> *"Draft and send an email to all students reminding them that the SQL Joins quiz is next Monday at 9am. Include what topics to study."*
- 5 runs | Real instructor `input()` per run (decisions: yes, no, yes, yes, yes)
- Routing check: `"email"`, `"approved"`, `"rejected"`, or `"not sent"` in response
- Latency includes human review and typing time (end-to-end measurement)

---

## 10. Results Summary

All 15 runs completed successfully with **100% routing accuracy**.

| Scenario | Runs | Avg Latency | Min | Max | Routing |
|---|---|---|---|---|---|
| Scenario 1 — Lesson Plan | 5 | 23.99s | 20.84s | 26.47s | 100% |
| Scenario 2 — Quiz Generation | 5 | 21.61s | 17.74s | 28.22s | 100% |
| Scenario 3 — Email HITL | 5 | 9.53s | 5.87s | 18.26s | 100% |
| **Overall** | **15** | **18.38s** | — | — | **100%** |

### Key Observations
- **Scenario 3 latency variance is high** (5.87–18.26s) — reflects real human reading+typing time, not LLM variance
- **Lesson Plan produced the longest responses** (6,874–9,623 chars) — significantly more verbose than ADK (3,776–4,652 chars)
- **Two-stage routing worked perfectly** — all 15 runs routed correctly on first attempt; LLM fallback was never needed
- **HITL edit path available** — instructors can paste modified drafts directly; this was not tested in the experiment but is architecturally supported
- **No `output_schema` + `tools` constraint** — quiz pipeline implemented cleanly without the 2-agent workaround required by ADK

---

## 11. Output Files

| File | Location | Description |
|---|---|---|
| `langgraph_metrics.csv` | `data/langgraph_metrics.csv` | Raw per-run metrics (15 rows) |
| `langgraph_latency_chart.png` | `output/langgraph_latency_chart.png` | Bar chart of avg latency by scenario |
| `01_langgraph_system.ipynb` | `notebooks/01_langgraph_system.ipynb` | Full experiment notebook with outputs |
