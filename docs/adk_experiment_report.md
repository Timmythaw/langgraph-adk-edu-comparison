# AI Teacher Assistant — Google ADK Implementation Report

> This document describes the actual implementation of the Google ADK experiment as executed in `notebooks/02_adk_system.ipynb`. It serves as the technical reference for the empirical comparison research at Mae Fah Luang University.

---

## 1. Overview

This notebook implements an AI Teaching Assistant using **Google Agent Development Kit (ADK) v1.27.1**, running on **Google Colab** with **Vertex AI** as the backend. It is one of two parallel implementations (LangGraph vs Google ADK) used for empirical comparison in the research paper.

### Research Context
- **Paper:** Comparative analysis of LangGraph vs Google ADK for Educational Context
- **Institution:** Mae Fah Luang University (MFU), Thailand
- **Evaluation:** 3 test scenarios, 5 runs each (15 total)
- **Goal:** Provide empirical findings for Section V of the research paper

### System Purpose
Automate three instructional tasks for university lecturers:
1. **Lesson Plan Generation** — grounded in course materials via Vertex AI Search
2. **Quiz Generation** — 10-question MCQ, output formatted for instructor review
3. **Student Email Drafting** — HITL-gated before sending

---

## 2. Environment & Dependencies

| Component | Value |
|---|---|
| Runtime | Google Colab (Python 3.12) |
| ADK Version | `google-adk 1.27.1` |
| LLM — Orchestrator | `gemini-2.5-pro` |
| LLM — Sub-agents | `gemini-2.5-flash` |
| RAG Backend | Vertex AI Search (Discovery Engine) |
| Google Cloud Project | `edu-teacher-assistant-prod` |
| Search Location | `global` |
| Datastore | `curriculum connector at GCS` |
| Vertex AI Init Location | `us-central1` |
| Auth | `google.colab.auth.authenticate_user()` |
| Vertex AI backend | `GOOGLE_GENAI_USE_VERTEXAI=1` |

---

## 3. Agent Architecture

The system uses ADK's **Agent-as-Tool pattern** with a root `LlmAgent` (orchestrator) routing to three specialist agents via `AgentTool` wrappers. Routing is **AutoFlow** — non-deterministic, driven by `gemini-2.5-pro` reading agent `description` fields.

```
RootOrchestrator (LlmAgent, gemini-2.5-pro)
│  ← AutoFlow routing
├── lesson_planner_agent (LlmAgent, gemini-2.5-flash)
│     └── retrieve_course_materials (Vertex AI Search)
├── quiz_generator_agent (SequentialAgent)
│     ├── quiz_content_agent   → output_key="quiz_questions_json"
│     └── quiz_publisher_agent → reads {quiz_questions_json}
└── email_agent (SequentialAgent)
      ├── email_drafter_agent  → output_key="email_draft"
      └── email_sender_agent   → HITL gate + confirmation
```

### Key ADK Constraints Observed

| Constraint | Impact |
|---|---|
| `output_schema` + `tools` cannot be combined | Quiz Generator required a 2-agent `SequentialAgent` workaround |
| AutoFlow routing is non-deterministic | Routing correctness measured across 5 identical runs per scenario |
| HITL is binary approve/reject only | Instructors cannot edit drafts during pause (unlike LangGraph) |
| `output_key` passes state between agents | Used for quiz JSON and email draft handoff |

---

## 4. RAG Tool — Vertex AI Search

`retrieve_course_materials(query)` queries the Vertex AI Search datastore using `discoveryengine.SearchServiceClient`, fetches the top 3 relevant snippets (`page_size=3`), and returns them as a `\n\n---\n\n`-joined string. Returns `"No relevant materials found."` if no snippets are available.

Verified during setup: successfully retrieved course material snippets for `"software testing lesson plan"`.

---

## 5. Sub-Agent Details

### Lesson Planner Agent
- **Type:** `LlmAgent` (`gemini-2.5-flash`)
- **Name:** `lesson_planner_agent`
- **Tools:** `retrieve_course_materials`
- **Description:** *"Creates detailed lesson plans aligned with course materials. Use this agent when the instructor requests a lesson plan, course outline, or lecture notes."*
- **Behavior:** Fetches course content via RAG, generates a structured 90-minute lesson plan including learning objectives, timing breakdown, teaching methods, assessment strategy, and required materials.

### Quiz Generator Agent
- **Type:** `SequentialAgent` (2-step)
- **Name:** `quiz_generator_agent`
- **Description:** *"Generates a 10-question multiple choice quiz from course materials. Use this agent when the instructor requests a quiz, test, or assessment."*
- **Why 2 agents:** ADK prevents combining `output_schema` and `tools` in the same `LlmAgent`. The workaround splits the task:
  1. `quiz_content_agent` — uses RAG tool, generates 10 MCQs as a JSON array, saved to session state via `output_key="quiz_questions_json"`
  2. `quiz_publisher_agent` — reads `{quiz_questions_json}` from state, formats into a clean numbered list with ✅ correct answers and explanations

### Email Agent
- **Type:** `SequentialAgent` (2-step)
- **Name:** `email_agent`
- **Description:** *"Drafts a student email, requests instructor approval via HITL, then confirms sending."*
  1. `email_drafter_agent` — drafts a formal email (`SUBJECT:` / `BODY:` format), saved to state via `output_key="email_draft"`
  2. `email_sender_agent` — reads `{email_draft}`, calls `request_instructor_approval()`, then responds with confirmation or rejection

### HITL Implementation
`request_instructor_approval(draft_subject, draft_body)` uses Python's `input()` to pause graph execution and display the draft to the instructor. Returns `"approved"` for input `"yes"`, `"rejected"` for anything else (including `"no"`). This is a **binary decision** — instructors can only approve or reject, not edit the draft in-place. In production, this mechanism integrates with the ADK UI layer.

---

## 6. Root Orchestrator

- **Type:** `LlmAgent` (`gemini-2.5-pro`)
- **Name:** `teacher_assistant_orchestrator`
- **Tools:** `AgentTool(lesson_planner_agent)`, `AgentTool(quiz_generator_agent)`, `AgentTool(email_agent)`
- **Routing:** AutoFlow — Gemini reads agent `description` fields to select the correct specialist agent

---

## 7. Runner Setup

```python
APP_NAME = "teacher_assistant_adk"
USER_ID  = "mfu_instructor_01"

session_service = InMemorySessionService()
runner = Runner(agent=root_orchestrator, app_name=APP_NAME, session_service=session_service)
```

`run_request(user_input)` creates a fresh UUID session per run via `session_service.create_session()`, sends the message with `runner.run_async()`, and returns the final response text from `event.content.parts[0].text`.

---

## 8. Metrics Logger

Per-run data recorded to `adk_metrics.csv`:

| Field | Description |
|---|---|
| `timestamp` | UTC ISO timestamp |
| `scenario` | Scenario name |
| `framework` | Always `"Google ADK"` |
| `routing_correct` | Boolean — correct agent invoked? |
| `latency_sec` | Wall-clock time to final response (includes HITL human input time) |
| `response_length` | Character count of response |
| `error` | Error message if run failed |

After all runs, the CSV is cleaned to retain only the last 5 rows per scenario (15 total) before download.

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
- 5 runs | Real instructor `input()` per run — decisions were: `[yes, no, yes, yes, yes]`
- All 5 runs logged under a single scenario label: `"Scenario 3 - Email HITL"` (no per-decision split)
- Routing check: `"email"`, `"approved"`, or `"rejected"` in response
- Latency includes real human review and typing time (end-to-end HITL measurement)

---

## 10. Results Summary

All 15 runs completed successfully with **100% routing accuracy**.

| Scenario | Runs | Avg Latency | Min | Max | Routing |
|---|---|---|---|---|---|
| Scenario 1 — Lesson Plan | 5 | 21.06s | 18.70s | 24.45s | 100% |
| Scenario 2 — Quiz Generation | 5 | 27.99s | 25.36s | 31.06s | 100% |
| Scenario 3 — Email HITL | 5 | 15.30s | 13.53s | 17.43s | 100% |
| **Overall** | **15** | **21.45s** | — | — | **100%** |

> Scenario 3 latency includes real human HITL review + typing time (end-to-end measurement). HITL decisions across the 5 runs were: `yes, no, yes, yes, yes`.

### Key Observations
- **Quiz Generation was the slowest** (27.99s avg) — higher token output for 10 structured MCQs plus 2-agent sequential overhead
- **Lesson Plan response lengths varied** (3,776–4,652 chars) — shorter than LangGraph's lesson plan output (6,874–9,623 chars)
- **`output_schema` + `tools` constraint required a 2-agent workaround** for Quiz Generation — adds architectural complexity vs LangGraph
- **HITL is binary only** — instructors cannot modify drafts mid-pause; decision is approve or reject only
- **AutoFlow routing worked reliably** — 100% accuracy across all 15 runs despite non-deterministic routing
- **`WARNING: non-text parts in response`** — a `google_genai.types` warning appeared on Scenario 1 Run 1 (`function_call` parts); this is benign and did not affect routing or output

---

## 11. Output Files

| File | Location | Description |
|---|---|---|
| `adk_metrics.csv` | Colab runtime (downloaded via `files.download`) | Raw per-run metrics (15 rows) |
| `adk_latency_chart.png` | Colab runtime (saved via `matplotlib`) | Bar chart of avg latency by scenario |
| `02_adk_system.ipynb` | `notebooks/02_adk_system.ipynb` | Full experiment notebook with outputs |
