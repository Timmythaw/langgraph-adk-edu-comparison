# AI Teacher Assistant — LangGraph Implementation Guide
> **For GitHub Copilot:** This document specifies the full LangGraph implementation of the AI Teacher Assistant system for empirical comparison research at Mae Fah Luang University. Follow these specifications exactly. All code patterns are sourced from the official LangGraph documentation at https://docs.langchain.com/oss/python/langgraph/overview

---

## 1. Project Overview

This system implements an AI Teaching Assistant using **LangGraph** — a low-level orchestration framework for building stateful, long-running agents. The system is one of two parallel implementations (LangGraph vs Google ADK) used for empirical comparison in a research paper.

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
| Orchestration | LangGraph (`langgraph`) | Research subject |
| LLM — Orchestrator | `gemini-2.5-pro` | High reasoning for routing |
| LLM — Sub-agents | `gemini-2.0-flash` | Fast generation |
| RAG | Vertex AI Search via LangChain | Matches ADK counterpart |
| Vector Store | FAISS (local testing) | No cloud dependency for dev |
| Checkpointer | `MemorySaver` (dev) / `PostgresSaver` (prod) | State persistence |
| Google APIs | `google-api-python-client` | Docs, Forms, Gmail |
| Observability | LangSmith (token logging) | Evaluation metrics |
| Package Manager | `uv` | Project standard |

---

## 3. Installation

```bash
uv init teacher-assistant-langgraph
cd teacher-assistant-langgraph

uv add langgraph
uv add langchain-google-vertexai
uv add langchain-community
uv add google-api-python-client google-auth-httplib2 google-auth-oauthlib
uv add faiss-cpu
uv add python-dotenv pydantic
uv add langsmith  # for token/latency logging
```

---

## 4. Project Structure

```
teacher-assistant-langgraph/
├── .env
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── state.py              # TypedDict state schema
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py     # Supervisor/orchestrator node
│   │   ├── lesson_planner.py # Lesson planner node + validator
│   │   ├── quiz_generator.py # Quiz generator node + validator
│   │   └── email_composer.py # Email composer node (HITL)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tool.py       # Vertex AI Search / FAISS retrieval
│   │   ├── docs_tool.py      # Google Docs API
│   │   ├── forms_tool.py     # Google Forms API
│   │   └── gmail_tool.py     # Gmail API
│   ├── graph.py              # Main graph assembly
│   ├── hitl.py               # Human-in-the-loop helpers
│   └── evaluation/
│       ├── __init__.py
│       ├── logger.py         # Metrics logger (latency, tokens, routing)
│       └── scenarios.py      # 3 test scenario definitions
├── tests/
│   ├── test_scenario_1.py
│   ├── test_scenario_2.py
│   └── test_scenario_3.py
└── main.py
```

---

## 5. State Schema (`src/state.py`)

LangGraph's core is a **typed state object** that persists throughout graph execution and is accessible to all nodes. Define it as a `TypedDict`.

> Source: https://docs.langchain.com/oss/python/langgraph/overview — "Human-in-the-loop: Incorporate human oversight by inspecting and modifying agent state at any point."

```python
# src/state.py
from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph.message import add_messages

class TeacherAssistantState(TypedDict):
    # Core request
    messages: Annotated[list, add_messages]   # conversation history
    request: str                               # original instructor request
    intent: Optional[str]                     # classified intent: "lesson_plan" | "quiz" | "email"

    # RAG context
    rag_context: Optional[str]                # retrieved document snippets
    rag_citations: Optional[list[str]]        # source document names

    # Generated content
    draft_output: Optional[str]               # current draft (lesson plan / quiz / email)
    structured_output: Optional[dict]         # parsed structured output (for quiz JSON)

    # Self-reflection / validation
    alignment_score: Optional[float]          # curriculum alignment score (0.0–1.0)
    validation_feedback: Optional[str]        # validator node feedback
    iteration_count: int                      # number of refinement loops

    # HITL
    awaiting_approval: bool                   # True when paused for instructor
    approval_status: Optional[Literal["approved", "rejected", "edited"]]
    instructor_edit: Optional[str]            # edited content from instructor

    # Output
    final_output: Optional[str]               # finalized output
    google_doc_url: Optional[str]
    google_form_url: Optional[str]
    email_sent: bool

    # Evaluation metrics (logged for research)
    total_tokens_used: int
    task_start_time: Optional[float]
    node_timings: dict                        # {node_name: elapsed_seconds}
```

---

## 6. Graph Architecture

```
[START]
    │
    ▼
[supervisor_node]  ←──────────────────────────────────────────┐
    │                                                          │
    ├─── intent="lesson_plan" ──▶ [lesson_planner_node]        │
    │                                  │                       │
    │                          [curriculum_validator_node]     │
    │                                  │                       │
    │                    alignment_score < 0.7 ────────────────┘ (loop back)
    │                    alignment_score >= 0.7
    │                                  │
    │                          [docs_creator_node] ──▶ [END]
    │
    ├─── intent="quiz" ─────▶ [quiz_generator_node]
    │                                  │
    │                          [quiz_validator_node]
    │                                  │
    │                          [forms_creator_node] ──▶ [END]
    │
    └─── intent="email" ────▶ [email_composer_node]
                                       │
                               [INTERRUPT: instructor approval]
                                       │
                              approved ├──▶ [gmail_send_node] ──▶ [END]
                              rejected └──▶ [email_composer_node] (retry)
```

---

## 7. Node Implementations

### 7.1 Supervisor Node (`src/agents/supervisor.py`)

The supervisor classifies intent and routes to the correct sub-agent. Uses deterministic Python routing (not LLM delegation) for governance compliance.

```python
# src/agents/supervisor.py
from langchain_google_vertexai import ChatVertexAI
from src.state import TeacherAssistantState

llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0.1)

SYSTEM_PROMPT = """You are a supervisor for an AI Teaching Assistant at Mae Fah Luang University.
Classify the instructor's request into exactly one category:
- "lesson_plan": Creating or updating lesson plans, course outlines, lecture notes
- "quiz": Creating assessments, quizzes, tests, exam questions
- "email": Drafting or sending emails, announcements, or communications to students

Respond with ONLY the category keyword. No explanation."""

def supervisor_node(state: TeacherAssistantState) -> dict:
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["request"]}
    ])
    intent = response.content.strip().lower()
    if intent not in ["lesson_plan", "quiz", "email"]:
        intent = "lesson_plan"  # safe fallback
    return {"intent": intent, "iteration_count": 0}

def route_by_intent(state: TeacherAssistantState) -> str:
    """Conditional edge function — deterministic routing based on state."""
    return state["intent"]
```

### 7.2 Lesson Planner Node (`src/agents/lesson_planner.py`)

```python
# src/agents/lesson_planner.py
import time
from langchain_google_vertexai import ChatVertexAI
from src.state import TeacherAssistantState
from src.tools.rag_tool import retrieve_course_materials

llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0.3)

LESSON_PLANNER_PROMPT = """You are an expert curriculum designer at Mae Fah Luang University.
Using the provided course materials as your primary source, create a detailed lesson plan.

Course Materials (RAG Context):
{rag_context}

Instructor Request:
{request}

Previous Feedback (if any):
{validation_feedback}

Create a structured lesson plan with:
- Learning Objectives (aligned with course syllabus)
- Duration and timing breakdown
- Teaching methods and activities
- Assessment strategy
- Required materials

Format as clean markdown."""

def lesson_planner_node(state: TeacherAssistantState) -> dict:
    start = time.time()

    # Retrieve RAG context if not already present
    rag_context = state.get("rag_context") or retrieve_course_materials(state["request"])

    prompt = LESSON_PLANNER_PROMPT.format(
        rag_context=rag_context,
        request=state["request"],
        validation_feedback=state.get("validation_feedback", "None")
    )

    response = llm.invoke(prompt)
    elapsed = time.time() - start

    timings = dict(state.get("node_timings", {}))
    timings["lesson_planner"] = elapsed

    return {
        "draft_output": response.content,
        "rag_context": rag_context,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "node_timings": timings,
        "total_tokens_used": state.get("total_tokens_used", 0) + (response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0)
    }
```

### 7.3 Curriculum Validator Node

```python
# Inside src/agents/lesson_planner.py (add below lesson_planner_node)
import re

VALIDATOR_PROMPT = """You are a curriculum quality validator.
Evaluate the lesson plan below against the course materials for:
1. Alignment with course learning objectives
2. Pedagogical soundness
3. Completeness

Course Materials:
{rag_context}

Lesson Plan:
{draft_output}

Respond in this exact format:
SCORE: [0.0-1.0]
FEEDBACK: [specific improvement instructions if score < 0.7, or "Approved" if >= 0.7]"""

validator_llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0.1)

def curriculum_validator_node(state: TeacherAssistantState) -> dict:
    prompt = VALIDATOR_PROMPT.format(
        rag_context=state.get("rag_context", "No materials provided"),
        draft_output=state.get("draft_output", "")
    )
    response = validator_llm.invoke(prompt)
    text = response.content

    score_match = re.search(r"SCORE:\s*([0-9.]+)", text)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL)

    score = float(score_match.group(1)) if score_match else 0.5
    feedback = feedback_match.group(1).strip() if feedback_match else "Needs revision"

    return {
        "alignment_score": score,
        "validation_feedback": feedback
    }

def route_after_validation(state: TeacherAssistantState) -> str:
    """
    Conditional edge: loop back if score too low or iteration limit not reached.
    Prevents infinite loops with max 3 iterations.
    """
    if state.get("alignment_score", 0) >= 0.7:
        return "approved"
    if state.get("iteration_count", 0) >= 3:
        return "approved"  # force through after 3 attempts
    return "retry"
```

### 7.4 Quiz Generator Node (`src/agents/quiz_generator.py`)

> **Important:** ADK has a known constraint — it cannot mix `output_schema` with `tools` in the same agent. LangGraph has no such limitation. This is a key measured difference in the evaluation.

```python
# src/agents/quiz_generator.py
import json
import time
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel
from src.state import TeacherAssistantState
from src.tools.rag_tool import retrieve_course_materials

class QuizQuestion(BaseModel):
    question: str
    options: list[str]       # 4 options
    correct_answer: int      # index 0-3
    explanation: str

class QuizOutput(BaseModel):
    title: str
    topic: str
    questions: list[QuizQuestion]

llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0.4)
structured_llm = llm.with_structured_output(QuizOutput)

QUIZ_PROMPT = """You are a quiz designer at Mae Fah Luang University.
Using ONLY the course materials below, generate quiz questions.

Course Materials (RAG Context):
{rag_context}

Instructor Request:
{request}

Generate exactly 10 multiple-choice questions. Each question must:
- Be directly answerable from the provided course materials
- Have exactly 4 options
- Include a brief explanation of the correct answer"""

def quiz_generator_node(state: TeacherAssistantState) -> dict:
    start = time.time()
    rag_context = state.get("rag_context") or retrieve_course_materials(state["request"])

    prompt = QUIZ_PROMPT.format(
        rag_context=rag_context,
        request=state["request"]
    )

    # LangGraph allows structured output AND tool use simultaneously
    quiz: QuizOutput = structured_llm.invoke(prompt)
    elapsed = time.time() - start

    timings = dict(state.get("node_timings", {}))
    timings["quiz_generator"] = elapsed

    return {
        "draft_output": quiz.model_dump_json(indent=2),
        "structured_output": quiz.model_dump(),
        "rag_context": rag_context,
        "node_timings": timings
    }
```

### 7.5 Email Composer + HITL Node (`src/agents/email_composer.py`)

This node demonstrates LangGraph's key HITL advantage: **stateful inspection with mid-session editing**.

```python
# src/agents/email_composer.py
import time
from langchain_google_vertexai import ChatVertexAI
from langgraph.types import interrupt     # official LangGraph interrupt
from src.state import TeacherAssistantState

llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0.3)

EMAIL_PROMPT = """You are a professional email assistant for a university lecturer.
Draft an email from the instructor to all students.

Request: {request}

The email must be:
- Professional and clear
- Appropriately formal for university communication
- Include subject line as first line prefixed with "Subject: "

Format:
Subject: [subject here]

[email body here]"""

def email_composer_node(state: TeacherAssistantState) -> dict:
    start = time.time()

    # If instructor provided an edit during HITL, use it as the draft
    if state.get("approval_status") == "edited" and state.get("instructor_edit"):
        draft = state["instructor_edit"]
    else:
        response = llm.invoke(EMAIL_PROMPT.format(request=state["request"]))
        draft = response.content

    elapsed = time.time() - start
    timings = dict(state.get("node_timings", {}))
    timings["email_composer"] = elapsed

    return {
        "draft_output": draft,
        "awaiting_approval": True,
        "approval_status": None,
        "node_timings": timings
    }

def email_hitl_node(state: TeacherAssistantState) -> dict:
    """
    HITL interrupt node. Pauses graph execution and exposes FULL STATE
    to instructor for review and optional editing.

    Unlike ADK's binary approve/reject, LangGraph allows the instructor
    to modify state.draft_output directly before resuming.
    This is the key measured difference in the oversight dimension.
    """
    # interrupt() pauses execution; the caller receives the current state
    # and can resume with updated state via graph.invoke(Command(resume=...))
    instructor_response = interrupt({
        "action": "review_email",
        "draft": state["draft_output"],
        "message": "Please review the email draft. You may approve, reject, or edit it.",
        "options": ["approve", "reject", "edit"]
    })

    # instructor_response is provided when graph is resumed
    return {
        "approval_status": instructor_response.get("decision"),
        "instructor_edit": instructor_response.get("edited_content"),
        "awaiting_approval": False
    }

def route_after_email_hitl(state: TeacherAssistantState) -> str:
    """Deterministic routing after HITL."""
    status = state.get("approval_status")
    if status == "approved":
        return "send"
    elif status == "edited":
        return "send"   # send the edited version
    else:
        return "retry"  # regenerate on rejection
```

---

## 8. Graph Assembly (`src/graph.py`)

```python
# src/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import TeacherAssistantState
from src.agents.supervisor import supervisor_node, route_by_intent
from src.agents.lesson_planner import (
    lesson_planner_node, curriculum_validator_node, route_after_validation
)
from src.agents.quiz_generator import quiz_generator_node
from src.agents.email_composer import (
    email_composer_node, email_hitl_node, route_after_email_hitl
)
from src.tools.docs_tool import docs_creator_node
from src.tools.forms_tool import forms_creator_node
from src.tools.gmail_tool import gmail_send_node

def build_graph():
    builder = StateGraph(TeacherAssistantState)

    # Add all nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("lesson_planner", lesson_planner_node)
    builder.add_node("curriculum_validator", curriculum_validator_node)
    builder.add_node("docs_creator", docs_creator_node)
    builder.add_node("quiz_generator", quiz_generator_node)
    builder.add_node("forms_creator", forms_creator_node)
    builder.add_node("email_composer", email_composer_node)
    builder.add_node("email_hitl", email_hitl_node)
    builder.add_node("gmail_send", gmail_send_node)

    # Entry point
    builder.add_edge(START, "supervisor")

    # Supervisor routing (conditional)
    builder.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "lesson_plan": "lesson_planner",
            "quiz": "quiz_generator",
            "email": "email_composer"
        }
    )

    # Lesson plan flow with self-reflection loop
    builder.add_edge("lesson_planner", "curriculum_validator")
    builder.add_conditional_edges(
        "curriculum_validator",
        route_after_validation,
        {
            "retry": "lesson_planner",   # loop back for refinement
            "approved": "docs_creator"
        }
    )
    builder.add_edge("docs_creator", END)

    # Quiz flow (no loop needed — single pass with structured output)
    builder.add_edge("quiz_generator", "forms_creator")
    builder.add_edge("forms_creator", END)

    # Email flow with HITL
    builder.add_edge("email_composer", "email_hitl")
    builder.add_conditional_edges(
        "email_hitl",
        route_after_email_hitl,
        {
            "send": "gmail_send",
            "retry": "email_composer"
        }
    )
    builder.add_edge("gmail_send", END)

    # Compile with MemorySaver checkpointer for state persistence
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

graph = build_graph()
```

---

## 9. HITL Resume Pattern (`src/hitl.py`)

> Source: LangGraph docs — "Human-in-the-loop: Incorporate human oversight by inspecting and modifying agent state at any point."

```python
# src/hitl.py
from langgraph.types import Command
from src.graph import graph

def run_with_hitl(initial_state: dict, config: dict) -> dict:
    """
    Runs the graph with HITL support.
    When the graph hits an interrupt(), this function returns the current state.
    The caller can then inspect and modify state, then call resume().
    """
    # Run until interrupt
    result = graph.invoke(initial_state, config=config)
    return result

def resume_with_approval(config: dict, decision: str, edited_content: str = None) -> dict:
    """
    Resume graph after instructor reviews the email.

    Args:
        config: thread config with thread_id for state retrieval
        decision: "approved", "rejected", or "edited"
        edited_content: provided only when decision == "edited"

    This is the KEY HITL advantage over ADK:
    - ADK: binary approve/reject only
    - LangGraph: instructor can pass back ANY state modification
    """
    resume_data = {"decision": decision}
    if edited_content:
        resume_data["edited_content"] = edited_content

    result = graph.invoke(
        Command(resume=resume_data),
        config=config
    )
    return result
```

---

## 10. RAG Tool (`src/tools/rag_tool.py`)

```python
# src/tools/rag_tool.py
# Uses FAISS locally for development; swap to Vertex AI Search for production
import os
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

COURSE_MATERIALS_DIR = "./data/course_materials"

_vectorstore = None

def _load_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    faiss_index_path = "./data/faiss_index"

    if os.path.exists(faiss_index_path):
        _vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Build index from course PDFs
        docs = []
        for filename in os.listdir(COURSE_MATERIALS_DIR):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(COURSE_MATERIALS_DIR, filename))
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        _vectorstore = FAISS.from_documents(chunks, embeddings)
        _vectorstore.save_local(faiss_index_path)

    return _vectorstore

def retrieve_course_materials(query: str, k: int = 5) -> str:
    """Retrieve relevant course material snippets for a given query."""
    store = _load_vectorstore()
    docs = store.similarity_search(query, k=k)
    context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])
    return context
```

---

## 11. Google Workspace Tools

### Google Docs (`src/tools/docs_tool.py`)
```python
# src/tools/docs_tool.py
import time
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from src.state import TeacherAssistantState

def get_docs_service():
    creds = Credentials.from_authorized_user_file("token.json", ["https://www.googleapis.com/auth/documents"])
    return build("docs", "v1", credentials=creds)

def docs_creator_node(state: TeacherAssistantState) -> dict:
    start = time.time()
    service = get_docs_service()

    # Create document
    doc = service.documents().create(body={"title": "AI-Generated Lesson Plan"}).execute()
    doc_id = doc["documentId"]

    # Insert content
    service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": state["draft_output"]}}]}
    ).execute()

    doc_url = f"https://docs.google.com/document/d/{doc_id}"
    elapsed = time.time() - start
    timings = dict(state.get("node_timings", {}))
    timings["docs_creator"] = elapsed

    return {"google_doc_url": doc_url, "final_output": state["draft_output"], "node_timings": timings}
```

### Google Forms (`src/tools/forms_tool.py`)
```python
# src/tools/forms_tool.py
import json, time
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from src.state import TeacherAssistantState

def get_forms_service():
    creds = Credentials.from_authorized_user_file("token.json", ["https://www.googleapis.com/auth/forms.body"])
    return build("forms", "v1", credentials=creds)

def forms_creator_node(state: TeacherAssistantState) -> dict:
    start = time.time()
    service = get_forms_service()
    quiz_data = state.get("structured_output", {})

    form = service.forms().create(body={"info": {"title": quiz_data.get("title", "Quiz")}}).execute()
    form_id = form["formId"]

    # Add questions
    requests = []
    for i, q in enumerate(quiz_data.get("questions", [])):
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
        service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()

    form_url = f"https://docs.google.com/forms/d/{form_id}"
    elapsed = time.time() - start
    timings = dict(state.get("node_timings", {}))
    timings["forms_creator"] = elapsed

    return {"google_form_url": form_url, "final_output": json.dumps(quiz_data), "node_timings": timings}
```

### Gmail (`src/tools/gmail_tool.py`)
```python
# src/tools/gmail_tool.py
import base64, time
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from src.state import TeacherAssistantState

def get_gmail_service():
    creds = Credentials.from_authorized_user_file("token.json", ["https://www.googleapis.com/auth/gmail.send"])
    return build("gmail", "v1", credentials=creds)

def gmail_send_node(state: TeacherAssistantState) -> dict:
    start = time.time()
    service = get_gmail_service()
    draft = state["draft_output"]

    lines = draft.split("\n")
    subject = lines[0].replace("Subject:", "").strip() if lines[0].startswith("Subject:") else "Class Announcement"
    body = "\n".join(lines[2:]) if len(lines) > 2 else draft

    message = MIMEText(body)
    message["to"] = "students@mfu.ac.th"   # replace with actual class list
    message["subject"] = subject
    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().messages().send(userId="me", body={"raw": encoded}).execute()
    elapsed = time.time() - start
    timings = dict(state.get("node_timings", {}))
    timings["gmail_send"] = elapsed

    return {"email_sent": True, "final_output": draft, "node_timings": timings}
```

---

## 12. Evaluation Logger (`src/evaluation/logger.py`)

```python
# src/evaluation/logger.py
import json, time, csv, os
from datetime import datetime
from src.state import TeacherAssistantState

METRICS_FILE = "./output/langgraph_metrics.csv"

FIELDNAMES = [
    "timestamp", "scenario", "framework", "intent",
    "total_latency_sec", "iteration_count", "total_tokens",
    "alignment_score", "routing_correct", "hitl_decision",
    "hitl_steps", "google_output_created", "error"
]

def log_run_metrics(scenario: str, state: TeacherAssistantState, routing_correct: bool, hitl_steps: int = 0, error: str = None):
    """Log all evaluation metrics for one test run."""
    os.makedirs("./output", exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)

    total_latency = sum(state.get("node_timings", {}).values())

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "scenario": scenario,
        "framework": "LangGraph",
        "intent": state.get("intent", "unknown"),
        "total_latency_sec": round(total_latency, 3),
        "iteration_count": state.get("iteration_count", 0),
        "total_tokens": state.get("total_tokens_used", 0),
        "alignment_score": state.get("alignment_score", "N/A"),
        "routing_correct": routing_correct,
        "hitl_decision": state.get("approval_status", "N/A"),
        "hitl_steps": hitl_steps,
        "google_output_created": bool(state.get("google_doc_url") or state.get("google_form_url") or state.get("email_sent")),
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

## 13. Test Scenarios (`src/evaluation/scenarios.py`)

```python
# src/evaluation/scenarios.py

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
        "third_run": "edit",
        "edited_content": "Subject: Reminder — SQL Joins Quiz on Monday\n\nDear Students,\n\nThis is a reminder that your SQL Joins quiz is scheduled for Monday at 9:00 AM.\n\nTopics to review: INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN.\n\nBest regards,\n[Instructor Name]"
    }
}

ALL_SCENARIOS = [SCENARIO_1, SCENARIO_2, SCENARIO_3]
```

---

## 14. Main Entry Point (`main.py`)

```python
# main.py
import uuid
from src.graph import graph
from src.evaluation.logger import log_run_metrics
from src.evaluation.scenarios import ALL_SCENARIOS, SCENARIO_3
from src.hitl import resume_with_approval

def run_scenario(scenario: dict, hitl_decision: str = "approve"):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "request": scenario["request"],
        "messages": [],
        "iteration_count": 0,
        "awaiting_approval": False,
        "email_sent": False,
        "total_tokens_used": 0,
        "node_timings": {}
    }

    try:
        result = graph.invoke(initial_state, config=config)

        # Handle HITL for email scenario
        if result.get("awaiting_approval"):
            hitl_steps = 1
            if hitl_decision == "edit":
                result = resume_with_approval(
                    config, "edited",
                    edited_content=scenario.get("hitl_simulation", {}).get("edited_content")
                )
            else:
                result = resume_with_approval(config, hitl_decision)
        else:
            hitl_steps = 0

        routing_correct = result.get("intent") == scenario["expected_intent"]
        log_run_metrics(scenario["name"], result, routing_correct, hitl_steps)
        return result

    except Exception as e:
        log_run_metrics(scenario["name"], {}, False, error=str(e))
        raise

if __name__ == "__main__":
    for scenario in ALL_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Running: {scenario['name']}")
        for i in range(scenario["runs"]):
            hitl_decision = "approve"
            if scenario.get("hitl_simulation"):
                decisions = ["approve", "reject", "edit", "approve", "approve"]
                hitl_decision = decisions[i % len(decisions)]
            result = run_scenario(scenario, hitl_decision=hitl_decision)
            print(f"  Run {i+1}: intent={result.get('intent')}, tokens={result.get('total_tokens_used')}")
```

---

## 15. Environment Variables (`.env`)

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=teacher-assistant-langgraph
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
```

---

## 16. Key Measurement Points for Research

| Dimension | What to Record | Where in Code |
|---|---|---|
| Control Flow | `routing_correct` (T/F per run) | `logger.py` |
| HITL Oversight | `hitl_steps`, `hitl_decision` | `logger.py` |
| Token Usage | `total_tokens_used` | `state.py` + each node |
| Latency | `node_timings` dict summed | each node function |
| Self-Reflection | `iteration_count`, `alignment_score` | `curriculum_validator_node` |
| Flexibility | Whether 3-step iterative edit works | HITL scenario 3 |
