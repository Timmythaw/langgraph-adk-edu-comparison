# AI Teacher Assistant — Google ADK Implementation Guide

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
| Auth | `google.colab.auth.authenticate_user()` |

---

## 3. Agent Architecture

The system uses ADK's **Agent-as-Tool pattern** with a root `LlmAgent` (orchestrator) routing to three specialist agents via `AgentTool` wrappers. Routing is **AutoFlow** — non-deterministic, driven by `gemini-2.5-pro` reading agent `description` fields.

```
RootOrchestrator (LlmAgent, gemini-2.5-pro)
│ ← AutoFlow routing
│ ├── lesson_planner_agent (LlmAgent, gemini-2.5-flash)
│ └── retrieve_course_materials (Vertex AI Search)
│ ├── quiz_generator_agent (SequentialAgent)
│ ├── quiz_content_agent (retrieves + generates JSON)
│ └── quiz_publisher_agent (formats + presents quiz)
│ └── email_agent (SequentialAgent)
├── email_drafter_agent (drafts email)
└── email_sender_agent (HITL gate + confirmation)
```


### Key ADK Constraints Observed

| Constraint | Impact |
|---|---|
| `output_schema` + `tools` cannot be combined | Quiz Generator required a 2-agent `SequentialAgent` workaround |
| AutoFlow routing is non-deterministic | Routing correctness measured across 5 identical runs per scenario |
| HITL is binary approve/reject only | Instructors cannot edit drafts during pause |
| `output_key` passes state between agents | Used for quiz JSON and email draft handoff |

---

## 4. RAG Tool — Vertex AI Search

`retrieve_course_materials(query)` queries the Vertex AI Search datastore, fetches the top 3 relevant snippets, and returns them as a concatenated string. Verified during setup: successfully retrieved course material snippets for `"software testing lesson plan"`.

---

## 5. Sub-Agent Details

### Lesson Planner Agent
- **Type:** `LlmAgent` (gemini-2.5-flash)
- **Tools:** `retrieve_course_materials`
- **Behavior:** Fetches course content via RAG, generates a structured 90-minute lesson plan with learning objectives, timing, teaching methods, assessment strategy, and materials.

### Quiz Generator Agent
- **Type:** `SequentialAgent` (2-step)
- **Why 2 agents:** ADK prevents combining `output_schema` and `tools` in the same agent. The workaround splits the task:
  1. `quiz_content_agent` — uses RAG tool, generates 10 MCQs as JSON, saved to state via `output_key="quiz_questions_json"`
  2. `quiz_publisher_agent` — reads `{quiz_questions_json}` from state, formats into a readable numbered list with ✅ correct answers

### Email Agent
- **Type:** `SequentialAgent` (2-step)
  1. `email_drafter_agent` — drafts the email, saved to state via `output_key="email_draft"`
  2. `email_sender_agent` — reads `{email_draft}`, calls `request_instructor_approval()`, then confirms

### HITL Implementation
`request_instructor_approval()` uses Python's `input()` to pause execution and prompt the instructor. Returns `"approved"` for `"yes"`, `"rejected"` for `"no"`. This simulates ADK's interrupt mechanism — in production, this integrates with the UI layer.

---

## 6. Root Orchestrator

- **Type:** `LlmAgent` (gemini-2.5-pro)
- **Tools:** `AgentTool` wrapping each sub-agent
- **Routing:** AutoFlow — Gemini reads agent descriptions to decide routing

---

## 7. Runner Setup

An `InMemorySessionService` and `Runner` initialized with `app_name="teacher_assistant_adk"` and `user_id="mfu_instructor_01"`. A `run_request(user_input)` helper creates a fresh UUID session per run and returns the final response text from `runner.run_async()`.

---

## 8. Metrics Logger

Per-run data recorded to `adk_metrics.csv`:

| Field | Description |
|---|---|
| `timestamp` | UTC ISO timestamp |
| `scenario` | Scenario name (includes HITL decision for Scenario 3) |
| `framework` | Always `"Google ADK"` |
| `routing_correct` | Boolean — correct agent invoked? |
| `latency_sec` | Wall-clock time to final response |
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
- 5 runs | Simulated decisions: `[yes, no, yes, yes, yes]`
- Routing check: `"email"`, `"approved"`, or `"rejected"` in response

---

## 10. Results Summary

All 15 runs completed successfully with **100% routing accuracy**.

| Scenario | Runs | Avg Latency | Min | Max | Routing |
|---|---|---|---|---|---|
| Scenario 1 — Lesson Plan | 5 | 25.73s | 20.02s | 33.99s | 100% |
| Scenario 2 — Quiz Generation | 5 | 33.11s | 30.81s | 34.67s | 100% |
| Scenario 3 — HITL (decision=no) | 1 | 28.43s | — | — | 100% |
| Scenario 3 — HITL (decision=yes) | 4 | 21.38s | 17.51s | 25.82s | 100% |
| **Overall** | **15** | **27.21s** | — | — | **100%** |

### Key Observations
- **Quiz Generation was the slowest** (33.11s avg) — higher token output for 10 structured MCQs
- **Lesson Plan had the widest variance** (20–34s) — response length varied from 3,656 to 5,957 chars
- **`output_schema` + `tools` constraint required a 2-agent workaround** for Quiz Generation — adds architectural complexity vs LangGraph
- **HITL is binary only** — instructors cannot modify drafts mid-pause, unlike LangGraph

---

## 11. Output Files

| File | Location | Description |
|---|---|---|
| `adk_metrics.csv` | `data/adk_metrics.csv` | Raw per-run metrics (15 rows) |
| `adk_latency_chart.png` | `output/adk_latency_chart.png` | Bar chart of avg latency by scenario |
| `02_adk_system.ipynb` | `notebooks/02_adk_system.ipynb` | Full experiment notebook with outputs |
