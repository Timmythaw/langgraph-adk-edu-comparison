# AI Teacher Assistant — Google ADK Implementation Report

> This document describes the actual implementation of the Google ADK experiment
> as executed in `notebooks/02_adk_system.ipynb`. It serves as the technical
> reference for the empirical comparison research at Mae Fah Luang University.

---

## 1. Overview

This notebook implements an AI Teaching Assistant using **Google Agent Development
Kit (ADK) v1.27.1**, running on **Google Colab** with **Vertex AI** as the backend.
It is one of two parallel implementations (LangGraph vs Google ADK) used for
empirical comparison in the research paper.

### Research Context
- **Paper:** Comparative analysis of LangGraph vs Google ADK for Educational Context
- **Institution:** Mae Fah Luang University (MFU), Thailand
- **Evaluation:** 3 test scenarios, 5 runs each (15 total) for latency; 3 runs per scenario for quality evaluation
- **Goal:** Provide empirical findings for the research paper

### System Purpose
Automate three instructional tasks for university lecturers:
1. **Lesson Plan Generation** — grounded in course materials via Vertex AI Search
2. **Quiz Generation** — 10-question MCQ, output formatted for instructor review
3. **Student Email Drafting** — HITL-gated via `before_tool_callback` before sending

---

## 2. Environment & Dependencies

| Component | Value |
|---|---|
| Runtime | Google Colab (Python 3.12) |
| ADK Version | `google-adk 1.27.1` |
| LLM — Orchestrator | `gemini-2.5-pro` (via `resilient_pro`) |
| LLM — Sub-agents | `gemini-2.5-flash` (via `resilient_flash`) |
| RAG Backend | Vertex AI Search (Discovery Engine) |
| Google Cloud Project | (Configured via Colab Secrets) |
| Search Location | `global` |
| Datastore | (GCS-backed, configured via Colab Secrets) |
| Vertex AI Init Location | `us-central1` |
| Auth | `google.colab.auth.authenticate_user()` |
| Vertex AI backend | `GOOGLE_GENAI_USE_VERTEXAI=1` |
| Tracing | LangSmith (`langsmith.integrations.google_adk.configure_google_adk()`) |
| LangSmith Project | `langgraph-adk-edu-comparison` |

---

## 3. Agent Architecture

The system uses ADK's **Agent-as-Tool pattern** with a root `LlmAgent`
(orchestrator) routing to three specialist agents via `AgentTool` wrappers.
Routing is **AutoFlow** — non-deterministic, driven by `gemini-2.5-pro` reading
agent `description` fields.

```
RootOrchestrator (LlmAgent, resilient_pro / gemini-2.5-pro)
│  ← AutoFlow routing
├── lesson_planner_agent (LlmAgent, resilient_flash)
│    └── retrieve_course_materials (Vertex AI Search)
├── quiz_generator_agent (SequentialAgent)
│    ├── quiz_content_agent → output_key="quiz_questions_json"
│    └── quiz_publisher_agent → reads {quiz_questions_json}
└── email_agent (SequentialAgent)
     ├── email_drafter_agent → output_key="email_draft"
     └── email_sender_agent → before_tool_callback HITL + send_email_to_students
```

### Key ADK Constraints Observed

| Constraint | Impact |
|---|---|
| `output_schema` + `tools` cannot be combined | Quiz Generator required a 2-agent `SequentialAgent` workaround |
| AutoFlow routing is non-deterministic | Routing correctness measured across 5 identical runs per scenario |
| HITL via `before_tool_callback` (binary) | Instructors cannot edit drafts during pause (unlike LangGraph) |
| `output_key` passes state between agents | Used for quiz JSON and email draft handoff |

---

## 4. Resilient Model Configuration

Both models are instantiated with a shared `HttpOptions` retry policy to handle
transient 429 / 5xx errors at the SDK level:

```python
resilient_http_options = types.HttpOptions(
    retry_options=types.HttpRetryOptions(
        attempts=5,
        initial_delay=5.0,
        http_status_codes=[408, 429, 500, 502, 503, 504]
    )
)

resilient_pro   = Gemini(model="gemini-2.5-pro",   http_options=resilient_http_options)
resilient_flash = Gemini(model="gemini-2.5-flash",  http_options=resilient_http_options)
```

---

## 5. RAG Tool — Vertex AI Search

`retrieve_course_materials(query, page_size)` queries the Vertex AI Search
datastore using `discoveryengine.SearchServiceClient`. The agent is instructed
to choose `page_size` based on task type:

| Task | Recommended `page_size` |
|---|---|
| Lesson Plan | 6–8 (rich detail needed) |
| Quiz | 6–8 (diverse content for 10 questions) |
| Email | 2–3 (light topic context only) |

Returns top snippets joined by `\n\n---\n\n`, or `"No relevant materials found."`.
Also decorated with `@traceable` for LangSmith retriever tracing.

---

## 6. Sub-Agent Details

### Lesson Planner Agent
- **Type:** `LlmAgent` (`resilient_flash`)
- **Name:** `lesson_planner_agent`
- **Tools:** `retrieve_course_materials`
- **Description:** *"Creates detailed lesson plans aligned with course materials. Use this agent when the instructor requests a lesson plan, course outline, or lecture notes."*
- **Behavior:** Fetches course content via RAG, generates a structured 90-minute lesson plan including learning objectives, timing breakdown, teaching methods, assessment strategy, and required materials.

### Quiz Generator Agent
- **Type:** `SequentialAgent` (2-step)
- **Name:** `quiz_generator_agent`
- **Description:** *"Generates a 10-question multiple choice quiz from course materials. Use this agent when the instructor requests a quiz, test, or assessment."*
- **Why 2 agents:** ADK prevents combining `output_schema` and `tools` in the same `LlmAgent`. The workaround splits the task:
  1. `quiz_content_agent` (`resilient_flash`) — uses RAG tool, generates 10 MCQs as a JSON array, saved to session state via `output_key="quiz_questions_json"`
  2. `quiz_publisher_agent` (`resilient_flash`) — reads `{quiz_questions_json}` from state, formats into a clean numbered list with ✅ correct answers and explanations

### Email Agent
- **Type:** `SequentialAgent` (2-step)
- **Name:** `email_agent`
- **Description:** *"Drafts a student email, requests instructor approval via HITL, then confirms sending."*
  1. `email_drafter_agent` (`resilient_flash`) — drafts a formal email (`SUBJECT:` / `BODY:` format), saved to state via `output_key="email_draft"`
  2. `email_sender_agent` (`resilient_flash`) — reads `{email_draft}`, triggers HITL via `before_tool_callback`, then calls `send_email_to_students(subject, body)`

### HITL Implementation
HITL is implemented via ADK's `before_tool_callback` on `email_sender_agent`.
`instructor_approval_callback(tool, args, tool_context)` intercepts the
`send_email_to_students` tool call, displays the draft subject and body to the
instructor, and calls `input()` to pause execution. Returns
`{"status": "rejected", ...}` for any non-`"yes"` input; returns `None` to
proceed with the actual tool call on approval. This is a **binary decision** —
instructors can only approve or reject, not edit the draft in-place.

---

## 7. Root Orchestrator

- **Type:** `LlmAgent` (`resilient_pro` / `gemini-2.5-pro`)
- **Name:** `teacher_assistant_orchestrator`
- **Tools:** `AgentTool(lesson_planner_agent)`, `AgentTool(quiz_generator_agent)`, `AgentTool(email_agent)`
- **Routing:** AutoFlow — Gemini reads agent `description` fields to select the correct specialist agent

---

## 8. Runner Setup

```python
APP_NAME = "teacher-assistant-adk"
USER_ID  = "mfu-instructor-01"

session_service = InMemorySessionService()
runner = Runner(agent=root_orchestrator, app_name=APP_NAME, session_service=session_service)
```

`run_request(user_input, scenario)` is decorated with `@traceable` for LangSmith
tracing. It creates a fresh UUID session per run, sends the message via
`runner.run_async()`, captures `usage_metadata` token counts from events, and
returns `(final_response, fallback_latency, input_tokens, output_tokens, ls_run_id)`.

Latency is sourced preferentially from LangSmith via `fetch_ls_latency(ls_run_id)`,
which polls for up to 30s before falling back to `time.time()` wall-clock.

---

## 9. Metrics Logger

Per-run data recorded to `adk_metrics.csv`:

| Field | Description |
|---|---|
| `scenario` | Scenario name |
| `latency_sec` | Latency in seconds (LangSmith-sourced or fallback) |
| `latency_source` | `"langsmith"` or `"fallback"` |
| `routing_correct` | Boolean — correct agent invoked? |
| `input_tokens` | Prompt token count from `usage_metadata` |
| `output_tokens` | Candidates token count from `usage_metadata` |
| `response_length` | Character count of response |
| `error` | Error message if run failed |

> Note: `timestamp` and `framework` fields are **not** present in the current notebook. `retry_count` is also not logged — retry resilience is handled at the SDK layer via `HttpRetryOptions`.

---

## 10. Experiment Scenarios

### Scenario 1 — Lesson Plan Generation
> *"Create a 90-minute lesson plan of week 1 on Software Testing for second-year Software Engineering students. Align it with the course materials."*
- 5 runs | Sleep: 60s between runs | Routing check: `"lesson plan"` or `"learning objectives"` in response

### Scenario 2 — Quiz Generation
> *"Generate 10 multiple-choice questions on Software Testing from the course materials."*
- 5 runs | Routing check: `"question"` or `"quiz"` in response

### Scenario 3 — Email with HITL
> *"Draft and send an email to all students reminding them that the SQL Joins quiz is next Monday at 9am. Include what topics to study."*
- 5 runs | Real instructor `input()` per run via `before_tool_callback`
- Routing check: `"email"`, `"approved"`, or `"rejected"` in response
- Latency is LangSmith end-to-end measurement (includes human review + typing time)

---

## 11. Results Summary

All 15 runs completed successfully with **100% routing accuracy**. All latency
values are LangSmith-sourced (100% `langsmith` source rate across all scenarios).

| Scenario | Runs | Avg Latency | Min | Max | Avg Input Tok | Avg Output Tok | Routing |
|---|---|---|---|---|---|---|---|
| Scenario 1 — Lesson Plan | 5 | 24.82s | 18.44s | 31.97s | 1363 | 926 | 100% |
| Scenario 2 — Quiz Generation | 5 | 26.39s | 19.21s | 32.27s | 1350 | 684 | 100% |
| Scenario 3 — Email HITL | 5 | 14.93s | 12.49s | 17.70s | 713 | 44 | 100% |
| **Overall** | **15** | **22.05s** | — | — | — | — | **100%** |

### Per-Run Detail

| Run | Scenario | Latency (s) | Source |
|---|---|---|---|
| 1 | Lesson Plan | 18.44 | langsmith |
| 2 | Lesson Plan | 19.85 | langsmith |
| 3 | Lesson Plan | 26.39 | langsmith |
| 4 | Lesson Plan | 31.97 | langsmith |
| 5 | Lesson Plan | 27.44 | langsmith |
| 6 | Quiz Generation | 24.13 | langsmith |
| 7 | Quiz Generation | 32.27 | langsmith |
| 8 | Quiz Generation | 19.21 | langsmith |
| 9 | Quiz Generation | 28.69 | langsmith |
| 10 | Quiz Generation | 27.63 | langsmith |
| 11 | Email HITL | 13.97 | langsmith |
| 12 | Email HITL | 15.32 | langsmith |
| 13 | Email HITL | 15.15 | langsmith |
| 14 | Email HITL | 12.49 | langsmith |
| 15 | Email HITL | 17.70 | langsmith |

### Key Observations
- **Quiz Generation was the slowest** (26.39s avg) due to the 2-agent SequentialAgent workaround
- **Email HITL was the fastest** (14.93s avg); HITL decision time is included but input() was answered promptly
- **`output_schema` + `tools` constraint required a 2-agent workaround** for Quiz Generation — adds architectural complexity vs LangGraph
- **HITL is binary only** via `before_tool_callback` — instructors cannot modify drafts mid-pause
- **AutoFlow routing worked reliably** — 100% accuracy across all 15 runs despite non-deterministic routing
- **All latency values are LangSmith-sourced** — 100% clean latency, no fallback measurements recorded
- **`WARNING: non-text parts in response`** — a `google_genai.types` warning appeared on Scenario 1 Run 1 (`function_call` parts); benign, did not affect routing or output

---

## 12. Quality Evaluation Results (LLM-as-a-Judge — Notebook 03)

3 runs per scenario evaluated by `gemini-2.5-pro` on 5 criteria (max 5 each, total 25).

| Scenario | Runs | Avg Total | Accuracy | Completeness | Clarity | Edu Relevance | Task Adherence |
|---|---|---|---|---|---|---|---|
| Lesson Plan | 3 | 22.33 | 5.0 | 4.67 | 5.0 | 4.67 | 3.00 |
| Quiz Generation | 3 | 23.00 | 5.0 | 5.00 | 5.0 | 3.00 | 5.00 |
| Email HITL | 3 | 19.67 | 5.0 | 3.67 | 4.0 | 3.33 | 3.67 |
| **Overall** | **9** | **21.67** | **5.0** | **4.44** | **4.67** | **3.67** | **3.89** |

> Email Run 3 returned only a send confirmation without the draft content, scoring 10/25 and pulling down the email average.

---

## 13. RAG Faithfulness (RAGAS — Notebook 04)

| Scenario | Faithfulness | Answer Relevancy | Context Precision | Contexts Retrieved |
|---|---|---|---|---|
| Lesson Plan | 0.705 | 0.815 | 0.000 | 2–6 chunks |
| Quiz | 0.037 | 0.739 | 0.667 | 2 chunks |
| Email | 0.000 | 0.434 | 0.000 | 0 chunks |

> Email runs retrieved 0 contexts (RAG was not triggered), resulting in faithfulness = 0.000.

---

## 14. Hallucination Analysis (Notebook 05)

| Scenario | Groundedness | RAGAS Faithfulness | Composite Halluc % |
|---|---|---|---|
| Lesson Plan | 1.0 | 0.705 | 43.2% |
| Quiz | 1.0 | 0.037 | 65.4% |
| Email | 1.0 | 0.000 | 66.7% |

---

## 15. Output Files

| File | Description |
|---|---|
| `adk_metrics.csv` | Raw per-run latency metrics (15 rows) |
| `adk_latency_chart.png` | Bar chart of avg latency by scenario with LangSmith source % |
| `judge_results.csv` | Per-run LLM Judge scores |
| `judge_radar_overall.png` | Radar chart comparing LangGraph vs ADK quality dimensions |
| `ragas_comparison_chart.png` | RAGAS metric comparison chart |
| `hallucination_results.csv` | Per-run hallucination scores |
| `hallucination_composite.csv` | Aggregated composite hallucination by framework and scenario |
| `hallucination_heatmap.png` | Heatmap of composite hallucination % |
| `02_adk_system.ipynb` | Full experiment notebook with cell outputs |
