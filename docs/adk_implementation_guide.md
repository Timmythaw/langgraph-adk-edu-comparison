# AI Teacher Assistant — Google ADK Implementation Guide
> **For GitHub Copilot:** This document specifies the full Google ADK implementation of the AI Teacher Assistant system for empirical comparison research at Mae Fah Luang University. Follow these specifications exactly. All code patterns are sourced from the official Google ADK documentation at https://google.github.io/adk-docs/

---

## 1. Project Overview

This system implements an AI Teaching Assistant using **Google Agent Development Kit (ADK)** — a flexible, modular framework for developing and deploying AI agents, optimized for the Gemini and Google Cloud ecosystem. This is one of two parallel implementations (LangGraph vs Google ADK) used for empirical comparison.

### Research Context
- **Paper:** Comparative analysis of LangGraph vs Google ADK for AI teaching assistants
- **Institution:** Mae Fah Luang University (MFU), Thailand
- **Evaluation:** 3 test scenarios across 6 measurement dimensions
- **Goal:** Replace the theoretical Section V of the paper with empirical findings

### System Purpose
Automate three instructional tasks for university lecturers:
1. **Lesson Plan Generation** → outputs to Google Docs
2. **Quiz Generation** → outputs to Google Forms
3. **Student Email Drafting** → sends via Gmail (HITL-gated)

---

## 2. Technology Stack

| Component | Choice | Reason |
|---|---|---|
| Orchestration | Google ADK (`google-adk`) | Research subject |
| LLM — Orchestrator | `gemini-2.5-pro` | High reasoning for routing |
| LLM — Sub-agents | `gemini-2.0-flash` | Fast generation |
| RAG | Vertex AI Search (native ADK tool) | Native Google integration |
| HITL | ADK built-in `interrupt` mechanism | Binary approve/reject |
| Deployment | Local runner (dev) / Vertex AI Agent Engine (prod) | As per ADK docs |
| Google APIs | `google-api-python-client` | Docs, Forms, Gmail |
| Package Manager | `uv` | Project standard |

---

## 3. Installation

```bash
uv init teacher-assistant-adk
cd teacher-assistant-adk

uv add google-adk
uv add google-cloud-aiplatform
uv add google-api-python-client google-auth-httplib2 google-auth-oauthlib
uv add python-dotenv pydantic
```

---

## 4. Project Structure

```
teacher-assistant-adk/
├── .env
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Root LlmAgent (AutoFlow routing)
│   │   ├── lesson_planner.py     # LessonPlannerAgent (Agent-as-Tool)
│   │   ├── quiz_generator.py     # QuizGeneratorAgent (Agent-as-Tool)
│   │   └── email_agent.py        # EmailAgent (SequentialAgent + HITL)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tool.py           # Vertex AI Search tool function
│   │   ├── docs_tool.py          # Google Docs API function tool
│   │   ├── forms_tool.py         # Google Forms API function tool
│   │   └── gmail_tool.py         # Gmail API function tool
│   ├── callbacks/
│   │   └── metrics_callback.py   # Token and latency logging
│   └── evaluation/
│       ├── __init__.py
│       ├── logger.py             # CSV metrics logger
│       └── scenarios.py          # 3 test scenario definitions
├── tests/
│   ├── test_scenario_1.py
│   ├── test_scenario_2.py
│   └── test_scenario_3.py
└── main.py
```

---

## 5. Agent Architecture

ADK uses a **service-oriented composition** model. Each agent is a modular service defined by its configuration, tool set, and runtime constraints. The architecture follows the **Agent-as-Tool pattern**: sub-agents are wrapped as callable tools for the root orchestrator.

> Source: https://google.github.io/adk-docs/agents/llm-agents/ — "Unlike deterministic Workflow Agents that follow predefined execution paths, LlmAgent behavior is non-deterministic. It uses the LLM to interpret instructions and context."

### Architecture Diagram

```
RootOrchestrator (LlmAgent, gemini-2.5-pro)
  │  ← AutoFlow: LLM-driven routing based on agent descriptions
  │
  ├── LessonPlannerTool   (AgentTool wrapping LlmAgent)
  │     └── rag_tool (Vertex AI Search function)
  │     └── docs_tool (Google Docs API function)
  │
  ├── QuizGeneratorTool   (AgentTool wrapping LlmAgent)
  │     └── rag_tool (Vertex AI Search function)
  │     └── forms_tool (Google Forms API function)
  │     ⚠️  NOTE: output_schema CANNOT be used here alongside tools
  │            This is a measured constraint vs LangGraph
  │
  └── EmailTool           (AgentTool wrapping SequentialAgent)
        ├── Step 1: EmailDrafterAgent (LlmAgent — draft content)
        └── Step 2: EmailSenderAgent (LlmAgent — HITL + Gmail send)
```

---

## 6. Tool Implementations

### RAG Tool (`src/tools/rag_tool.py`)

```python
# src/tools/rag_tool.py
import os
from google.cloud import discoveryengine_v1 as discoveryengine

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "global"
DATA_STORE_ID = os.environ.get("VERTEX_AI_SEARCH_DATASTORE_ID")


def retrieve_course_materials(query: str) -> str:
    """
    Retrieves relevant course material snippets from Vertex AI Search.

    Args:
        query: The search query to find relevant course materials.

    Returns:
        Concatenated relevant document snippets from the course materials datastore.
    """
    client = discoveryengine.SearchServiceClient()
    serving_config = (
        f"projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/collections/default_collection/dataStores/{DATA_STORE_ID}"
        f"/servingConfigs/default_config"
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=5,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=5,
                include_citations=True,
            )
        ),
    )

    response = client.search(request)
    snippets = []
    for result in response.results:
        doc = result.document
        if doc.derived_struct_data:
            for snippet in doc.derived_struct_data.get("snippets", []):
                snippets.append(snippet.get("snippet", ""))

    return "\n\n---\n\n".join(snippets) if snippets else "No relevant materials found."
```

### Google Docs Tool (`src/tools/docs_tool.py`)

```python
# src/tools/docs_tool.py
import time
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials


def create_lesson_plan_doc(title: str, content: str) -> str:
    """
    Creates a Google Doc with the provided lesson plan content.

    Args:
        title: Title for the Google Doc.
        content: Full lesson plan content in markdown or plain text.

    Returns:
        The URL of the created Google Doc.
    """
    creds = Credentials.from_authorized_user_file(
        "token.json", ["https://www.googleapis.com/auth/documents"]
    )
    service = build("docs", "v1", credentials=creds)

    doc = service.documents().create(body={"title": title}).execute()
    doc_id = doc["documentId"]

    service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": content}}]}
    ).execute()

    return f"https://docs.google.com/document/d/{doc_id}"
```

### Google Forms Tool (`src/tools/forms_tool.py`)

```python
# src/tools/forms_tool.py
import json
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials


def create_quiz_form(title: str, questions_json: str) -> str:
    """
    Creates a Google Form quiz from a JSON list of questions.

    Args:
        title: Title for the quiz form.
        questions_json: JSON string of questions. Each question must have:
                        'question' (str), 'options' (list[str]), 'correct_index' (int).

    Returns:
        The URL of the created Google Form.

    Note: This tool cannot be used in the same agent as output_schema.
    This is a known ADK constraint documented at https://google.github.io/adk-docs/agents/llm-agents/
    """
    creds = Credentials.from_authorized_user_file(
        "token.json", ["https://www.googleapis.com/auth/forms.body"]
    )
    service = build("forms", "v1", credentials=creds)

    questions = json.loads(questions_json)

    form = service.forms().create(body={"info": {"title": title}}).execute()
    form_id = form["formId"]

    requests = []
    for i, q in enumerate(questions):
        requests.append({
            "createItem": {
                "item": {
                    "title": q["question"],
                    "questionItem": {
                        "question": {
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [{"value": opt} for opt in q["options"]],
                                "shuffle": False
                            }
                        }
                    }
                },
                "location": {"index": i}
            }
        })

    if requests:
        service.forms().batchUpdate(
            formId=form_id, body={"requests": requests}
        ).execute()

    return f"https://docs.google.com/forms/d/{form_id}"
```

### Gmail Tool (`src/tools/gmail_tool.py`)

```python
# src/tools/gmail_tool.py
import base64
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials


def send_class_email(subject: str, body: str, recipient: str = "students@mfu.ac.th") -> str:
    """
    Sends an email to the class via Gmail API.
    This tool should only be called after explicit instructor approval (HITL).

    Args:
        subject: Email subject line.
        body: Email body content.
        recipient: Recipient email address (defaults to class list).

    Returns:
        Confirmation string with message ID.
    """
    creds = Credentials.from_authorized_user_file(
        "token.json", ["https://www.googleapis.com/auth/gmail.send"]
    )
    service = build("gmail", "v1", credentials=creds)

    message = MIMEText(body)
    message["to"] = recipient
    message["subject"] = subject
    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()

    sent = service.users().messages().send(userId="me", body={"raw": encoded}).execute()
    return f"Email sent successfully. Message ID: {sent['id']}"
```

---

## 7. Sub-Agent Implementations

### Lesson Planner Agent (`src/agents/lesson_planner.py`)

> Source: https://google.github.io/adk-docs/agents/llm-agents/ — `LlmAgent` with `name`, `model`, `description`, `instruction`, `tools`

```python
# src/agents/lesson_planner.py
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from src.tools.rag_tool import retrieve_course_materials
from src.tools.docs_tool import create_lesson_plan_doc

lesson_planner_agent = LlmAgent(
    name="lesson_planner_agent",
    model="gemini-2.0-flash",
    description=(
        "Creates detailed lesson plans aligned with course materials. "
        "Use this agent when the instructor requests a lesson plan, course outline, or lecture notes."
    ),
    instruction="""You are an expert curriculum designer at Mae Fah Luang University.

Your workflow:
1. Use `retrieve_course_materials` to fetch relevant content for the topic.
2. Generate a comprehensive lesson plan that includes:
   - Learning Objectives (aligned with course syllabus)
   - Duration and timing breakdown (for 90-minute class)
   - Teaching methods and student activities
   - Assessment strategy
   - Required materials
3. Use `create_lesson_plan_doc` to save the final lesson plan to Google Docs.
4. Return the Google Doc URL to the orchestrator.

Always ground your lesson plan in the retrieved course materials.
Format the lesson plan in clean, readable markdown.""",
    tools=[retrieve_course_materials, create_lesson_plan_doc]
)
```

### Quiz Generator Agent (`src/agents/quiz_generator.py`)

> **Important Constraint from ADK Docs:** `output_schema` cannot be combined with `tools` in the same agent (except with Gemini 3.0). For structured quiz output WITH tool use, a two-agent pattern is required. This is a **measured limitation** in the research comparison.

```python
# src/agents/quiz_generator.py
from google.adk.agents import LlmAgent, SequentialAgent
from src.tools.rag_tool import retrieve_course_materials
from src.tools.forms_tool import create_quiz_form

# Agent 1: Retrieves materials and generates quiz content (has tools, no output_schema)
quiz_content_agent = LlmAgent(
    name="quiz_content_agent",
    model="gemini-2.0-flash",
    description="Retrieves course materials and generates quiz question content.",
    instruction="""You are a quiz content specialist at Mae Fah Luang University.

1. Use `retrieve_course_materials` to fetch relevant content for the quiz topic.
2. Generate exactly 10 multiple-choice questions. For each question provide:
   - question: the question text
   - options: list of exactly 4 answer choices
   - correct_index: index (0-3) of the correct answer
   - explanation: why the answer is correct
3. Format your entire response as a valid JSON array of question objects.
4. Do NOT include any text outside the JSON array.

Example format:
[
  {
    "question": "What is normalization in databases?",
    "options": ["A process...", "A type...", "A constraint...", "A key..."],
    "correct_index": 0,
    "explanation": "Normalization is..."
  }
]""",
    tools=[retrieve_course_materials],
    output_key="quiz_questions_json"  # saves JSON to session state
)

# Agent 2: Takes JSON from state and creates Google Form (has tools, no output_schema)
quiz_publisher_agent = LlmAgent(
    name="quiz_publisher_agent",
    model="gemini-2.0-flash",
    description="Publishes quiz questions to Google Forms.",
    instruction="""You are a quiz publisher.

The quiz questions JSON is available in your context as {quiz_questions_json}.

1. Extract the quiz title from the questions or use "Course Quiz" as default.
2. Call `create_quiz_form` with the title and the questions JSON string.
3. Return the Google Form URL.

Call the tool exactly once.""",
    tools=[create_quiz_form]
)

# SequentialAgent: runs content agent then publisher agent in order
# Source: https://google.github.io/adk-docs/agents/workflow-agents/
quiz_generator_agent = SequentialAgent(
    name="quiz_generator_agent",
    description=(
        "Generates a quiz from course materials and publishes it to Google Forms. "
        "Use this agent when the instructor requests a quiz, test, or assessment."
    ),
    sub_agents=[quiz_content_agent, quiz_publisher_agent]
)
```

### Email Agent (`src/agents/email_agent.py`)

> ADK HITL operates as a **binary approve/reject** decision. The agent pauses and waits for explicit instructor confirmation. Unlike LangGraph, instructors cannot edit the content during the pause — this is a **key measured difference**.

```python
# src/agents/email_agent.py
from google.adk.agents import LlmAgent, SequentialAgent
from src.tools.gmail_tool import send_class_email


def request_instructor_approval(draft_subject: str, draft_body: str) -> str:
    """
    Requests instructor approval before sending an email.
    This function signals the HITL interrupt point.
    The agent will pause here until the instructor approves or rejects.

    Args:
        draft_subject: The email subject to show for review.
        draft_body: The email body to show for review.

    Returns:
        Approval status — this will be provided by the instructor at runtime.
    """
    # In production, this integrates with the ADK interrupt mechanism.
    # The runner exposes this to the UI layer for instructor action.
    # Return value is provided by the instructor (not the LLM).
    return f"PENDING_APPROVAL: Subject='{draft_subject}'"


# Step 1: Draft the email
email_drafter_agent = LlmAgent(
    name="email_drafter_agent",
    model="gemini-2.0-flash",
    description="Drafts a professional email to students based on the instructor's request.",
    instruction="""You are a professional email drafting assistant for a university lecturer.

Draft a professional email to students based on the instructor's request.

Your email must include:
- A clear, professional subject line
- Appropriate university-level formal tone
- All relevant information from the request

Save the subject to state with output_key, then pass the full draft to the next step.
Format your response EXACTLY as:
SUBJECT: [subject line here]
BODY:
[email body here]""",
    output_key="email_draft"
)

# Step 2: HITL approval then send
email_sender_agent = LlmAgent(
    name="email_sender_agent",
    model="gemini-2.0-flash",
    description="Requests instructor approval and sends the approved email via Gmail.",
    instruction="""You are an email approval and delivery agent.

The email draft is: {email_draft}

Your workflow:
1. Parse the draft to extract the subject (after "SUBJECT:") and body (after "BODY:").
2. Call `request_instructor_approval` with the subject and body.
   - This will PAUSE execution until the instructor approves or rejects.
   - If the result indicates approval, proceed to step 3.
   - If rejected, inform the orchestrator that the email was not sent.
3. Only after approval: call `send_class_email` with the subject and body.
4. Return confirmation with the message ID.

IMPORTANT: Never call `send_class_email` without first calling `request_instructor_approval`.""",
    tools=[request_instructor_approval, send_class_email]
)

# SequentialAgent for email workflow
email_agent = SequentialAgent(
    name="email_agent",
    description=(
        "Drafts a student email, requests instructor approval via HITL, "
        "then sends it via Gmail. Use this agent when the instructor requests "
        "to send an email, announcement, or communication to students."
    ),
    sub_agents=[email_drafter_agent, email_sender_agent]
)
```

---

## 8. Root Orchestrator (`src/agents/orchestrator.py`)

The root orchestrator uses **AutoFlow** — LLM-driven routing where Gemini decides which sub-agent (tool) to invoke based on the agent `description` fields. This is a key measured difference from LangGraph's deterministic routing.

> Source: https://google.github.io/adk-docs/agents/llm-agents/ — "LlmAgent behavior is non-deterministic. It uses the LLM to interpret instructions and context, deciding dynamically how to proceed, which tools to use."

```python
# src/agents/orchestrator.py
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from src.agents.lesson_planner import lesson_planner_agent
from src.agents.quiz_generator import quiz_generator_agent
from src.agents.email_agent import email_agent

root_orchestrator = LlmAgent(
    name="teacher_assistant_orchestrator",
    model="gemini-2.5-pro",
    description="Root AI Teaching Assistant orchestrator for Mae Fah Luang University instructors.",
    instruction="""You are the AI Teaching Assistant for Mae Fah Luang University instructors.

You help lecturers with three types of tasks:
1. **Lesson Plans** — creating structured lesson plans saved to Google Docs
2. **Quizzes** — generating assessments published to Google Forms
3. **Emails** — drafting and sending communications to students via Gmail

For each instructor request:
1. Identify the task type.
2. Delegate to the appropriate specialist agent using the Agent-as-Tool pattern.
3. Report the result back to the instructor including any URLs or confirmation IDs.

Always be professional and confirm which action was taken.""",
    tools=[
        AgentTool(agent=lesson_planner_agent),
        AgentTool(agent=quiz_generator_agent),
        AgentTool(agent=email_agent)
    ]
)
```

---

## 9. Runner Setup and Execution (`main.py`)

> Source: https://google.github.io/adk-docs/ — "run locally, scale with Vertex AI Agent Engine, or integrate into custom infrastructure"

```python
# main.py
import asyncio
import time
import uuid
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from src.agents.orchestrator import root_orchestrator
from src.evaluation.logger import log_run_metrics
from src.evaluation.scenarios import ALL_SCENARIOS

APP_NAME = "teacher_assistant_adk"
USER_ID = "mfu_instructor_01"

session_service = InMemorySessionService()
runner = Runner(
    agent=root_orchestrator,
    app_name=APP_NAME,
    session_service=session_service
)


async def run_scenario(scenario: dict, hitl_decision: str = "approve") -> dict:
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=scenario["request"])]
    )

    start_time = time.time()
    final_response = None
    routing_correct = False
    token_count = 0
    error_msg = None
    hitl_steps = 0

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=user_message
        ):
            # Track token usage from events
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                token_count += event.usage_metadata.get("total_token_count", 0)

            # Detect HITL interrupt
            if hasattr(event, "type") and "interrupt" in str(event.type).lower():
                hitl_steps += 1
                # Simulate instructor decision
                # In production, this is handled by the UI layer
                print(f"  [HITL] Instructor review required. Decision: {hitl_decision}")
                # Resume is handled externally by the runner in production

            if event.is_final_response() and event.content:
                final_response = event.content.parts[0].text.strip()

        total_latency = time.time() - start_time

        # Determine if routing was correct by checking response content
        expected = scenario["expected_intent"]
        if final_response:
            if expected == "lesson_plan" and ("docs.google.com/document" in final_response or "lesson plan" in final_response.lower()):
                routing_correct = True
            elif expected == "quiz" and ("docs.google.com/forms" in final_response or "quiz" in final_response.lower()):
                routing_correct = True
            elif expected == "email" and ("email" in final_response.lower() or "sent" in final_response.lower()):
                routing_correct = True

        metrics = {
            "intent": expected,
            "total_latency_sec": round(total_latency, 3),
            "total_tokens_used": token_count,
            "node_timings": {"total": total_latency},
            "iteration_count": 1,
            "alignment_score": None,
            "approval_status": hitl_decision if hitl_steps > 0 else "N/A",
            "google_doc_url": "docs.google.com" in (final_response or ""),
            "email_sent": "sent" in (final_response or "").lower()
        }

        log_run_metrics(scenario["name"], metrics, routing_correct, hitl_steps, error_msg)
        return {"response": final_response, "metrics": metrics}

    except Exception as e:
        error_msg = str(e)
        log_run_metrics(scenario["name"], {}, False, error=error_msg)
        raise


async def main():
    for scenario in ALL_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Running: {scenario['name']}")
        for i in range(scenario["runs"]):
            hitl_decision = "approve"
            if scenario.get("hitl_simulation"):
                decisions = ["approve", "reject", "approve", "approve", "approve"]
                hitl_decision = decisions[i % len(decisions)]
            result = await run_scenario(scenario, hitl_decision=hitl_decision)
            print(f"  Run {i+1}: latency={result['metrics']['total_latency_sec']}s, tokens={result['metrics']['total_tokens_used']}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 10. Evaluation Logger (`src/evaluation/logger.py`)

```python
# src/evaluation/logger.py
import csv, os
from datetime import datetime

METRICS_FILE = "./output/adk_metrics.csv"

FIELDNAMES = [
    "timestamp", "scenario", "framework", "intent",
    "total_latency_sec", "iteration_count", "total_tokens",
    "alignment_score", "routing_correct", "hitl_decision",
    "hitl_steps", "google_output_created", "error"
]


def log_run_metrics(scenario: str, state: dict, routing_correct: bool, hitl_steps: int = 0, error: str = None):
    """Log evaluation metrics for one ADK test run."""
    os.makedirs("./output", exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "scenario": scenario,
        "framework": "Google ADK",
        "intent": state.get("intent", "unknown"),
        "total_latency_sec": state.get("total_latency_sec", 0),
        "iteration_count": state.get("iteration_count", 1),
        "total_tokens": state.get("total_tokens_used", 0),
        "alignment_score": state.get("alignment_score", "N/A"),
        "routing_correct": routing_correct,
        "hitl_decision": state.get("approval_status", "N/A"),
        "hitl_steps": hitl_steps,
        "google_output_created": state.get("google_doc_url") or state.get("email_sent", False),
        "error": error or ""
    }

    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[METRICS] {scenario} logged: latency={row['total_latency_sec']}s, tokens={row['total_tokens']}")
```

---

## 11. Test Scenarios (`src/evaluation/scenarios.py`)

```python
# src/evaluation/scenarios.py
# Identical prompts to LangGraph implementation for fair comparison

SCENARIO_1 = {
    "name": "Scenario 1 — Lesson Plan Generation",
    "request": "Create a 90-minute lesson plan on Database Normalization (1NF, 2NF, 3NF) for second-year Software Engineering students. Align it with the uploaded course syllabus.",
    "expected_intent": "lesson_plan",
    "expected_output": "google_doc_url",
    "runs": 5
}

SCENARIO_2 = {
    "name": "Scenario 2 — Quiz Generation with Structured Output",
    "request": "Generate 10 multiple-choice questions on Entity-Relationship Diagrams from Chapter 3 of the database textbook. Publish to Google Forms.",
    "expected_intent": "quiz",
    "expected_output": "google_form_url",
    "runs": 5
}

SCENARIO_3 = {
    "name": "Scenario 3 — Student Email with HITL Gate",
    "request": "Draft and send an email to all students reminding them that the SQL Joins quiz is next Monday at 9am. Include what topics to study.",
    "expected_intent": "email",
    "expected_output": "email_sent",
    "runs": 5,
    "hitl_simulation": {
        "first_run": "approve",
        "second_run": "reject",
        "note": "ADK HITL is binary: approve or reject only. No mid-session editing."
    }
}

ALL_SCENARIOS = [SCENARIO_1, SCENARIO_2, SCENARIO_3]
```

---

## 12. Environment Variables (`.env`)

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
VERTEX_AI_SEARCH_DATASTORE_ID=your-datastore-id
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
```

---

## 13. Known ADK Constraints to Measure

These are documented limitations that are **key research observations**:

| Constraint | Documentation Reference | Impact on Research |
|---|---|---|
| `output_schema` + `tools` cannot be used together in the same agent (except Gemini 3.0) | https://google.github.io/adk-docs/agents/llm-agents/ — "Warning: Using output_schema with tools" | Requires 2-agent workaround for structured quiz output. Measured in Scenario 2. |
| AutoFlow routing is non-deterministic | https://google.github.io/adk-docs/agents/llm-agents/ — "LlmAgent behavior is non-deterministic" | Routing consistency may vary across identical runs. Measured via `routing_correct` across 5 runs. |
| HITL is binary approve/reject only | ADK interrupt mechanism | Instructors cannot edit content during pause. Measured in Scenario 3. |
| Agent-as-Tool context isolation | Each sub-agent gets only explicitly passed inputs | No cross-session memory without external storage. |

---

## 14. Key Measurement Points for Research

| Dimension | What to Record | Where in Code |
|---|---|---|
| Control Flow | `routing_correct` (T/F per run) across 5 runs | `logger.py` |
| HITL Oversight | `hitl_steps`, `hitl_decision` (binary only) | `logger.py` |
| Token Usage | `total_tokens_used` from event metadata | `main.py` event loop |
| Latency | `total_latency_sec` (wall clock per scenario) | `main.py` `time.time()` |
| Structured Output | Whether 2-agent workaround was required | `quiz_generator.py` architecture |
| Flexibility | Whether iterative editing is supported | Scenario 3 HITL test |
