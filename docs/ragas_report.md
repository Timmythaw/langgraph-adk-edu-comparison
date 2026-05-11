# RAGAS RAG Faithfulness Evaluation Report
Generated: 2026-05-09T06:10:54.199137+00:00

## Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Output grounded in retrieved chunks? |
| Answer Relevancy | Output relevant to the prompt? |
| Context Precision | Retrieved the right chunks? |
| Context Recall | Retrieved all needed chunks? |

## Overall Comparison

```
           faithfulness        answer_relevancy        context_precision        context_recall      latency        
                   mean    std             mean    std              mean    std           mean  std    mean     std
framework                                                                                                          
Google ADK        0.278  0.433            0.663  0.261             0.222  0.441            0.0  0.0   48.64  66.240
LangGraph         0.396  0.429            0.675  0.110             0.000  0.000            0.0  0.0   43.89  77.876
```

## Per-Scenario Breakdown

```
                        faithfulness  answer_relevancy  context_precision  context_recall  latency  n_contexts
framework  scenario                                                                                           
Google ADK email               0.000             0.434              0.000             0.0   82.273         1.0
           lesson_plan         0.705             0.815              0.000             0.0   32.880         4.0
           quiz                0.037             0.739              0.667             0.0   30.767         2.0
LangGraph  email               0.110             0.581              0.000             0.0    7.163         2.0
           lesson_plan         0.964             0.797              0.000             0.0   98.127         2.0
           quiz                0.115             0.647              0.000             0.0   26.380         2.0
```

## Context Chunks Captured

### LangGraph | lesson_plan
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are <b>two</b> major types of <b>software testing</b> - Black box testing : focuses on inp...
- Chunk 2: Measures &amp; Metrics Process Source: Roger S. Pressman, “<b>Software Engineering</b> – A practitioner&#39;s Approach”,...

### LangGraph | lesson_plan
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are <b>two</b> major types of <b>software testing</b> - Black box testing : focuses on inp...
- Chunk 2: Measures &amp; Metrics Process Source: Roger S. Pressman, “<b>Software Engineering</b> – A practitioner&#39;s Approach”,...

### LangGraph | lesson_plan
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are <b>two</b> major types of <b>software testing</b> - Black box testing : focuses on inp...
- Chunk 2: Measures &amp; Metrics Process Source: Roger S. Pressman, “<b>Software Engineering</b> – A practitioner&#39;s Approach”,...

### Google ADK | lesson_plan
- Contexts retrieved: 6
- Chunk 1: <b>Software Testing</b> There are <b>two</b> major types of <b>software testing</b> - Black box testing : focuses on inp...
- Chunk 2: SQA area <b>1</b>: SQA process implementation activities a. Establishing SQA processes and their coordination with the <...
- Chunk 3: <b>Software Testing</b> There are two major types of <b>software testing</b> - Black box <b>testing</b> : focuses on inp...

### Google ADK | lesson_plan
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are two major <b>types</b> of <b>software testing</b> - Black box testing : focuses on inp...
- Chunk 2: Common <b>types</b> of <b>software</b> configuration items: design documents • <b>Software</b> development plan (SDP) • ...

### Google ADK | lesson_plan
- Contexts retrieved: 4
- Chunk 1: <b>Software Testing</b> There are <b>two</b> major <b>types</b> of <b>software testing</b> - Black box <b>testing</b> : ...
- Chunk 2: Measures &amp; Metrics <b>Process</b> Source: Roger S. Pressman, “<b>Software Engineering</b> – A practitioner&#39;s App...
- Chunk 3: Unit <b>Test</b> • Unit <b>Test</b> is a level of the <b>software testing process</b> where individual units/components ...

### LangGraph | quiz
- Contexts retrieved: 2
- Chunk 1: • Automation Testing is use of Tools or Software to perform <b>Software Testing</b>. • Developing and executing tests th...
- Chunk 2: Establish SQA processes Coordination with relevant <b>software</b> processes....

### LangGraph | quiz
- Contexts retrieved: 2
- Chunk 1: • Automation Testing is use of Tools or Software to perform <b>Software Testing</b>. • Developing and executing tests th...
- Chunk 2: Establish SQA processes Coordination with relevant <b>software</b> processes....

### LangGraph | quiz
- Contexts retrieved: 2
- Chunk 1: • Automation Testing is use of Tools or Software to perform <b>Software Testing</b>. • Developing and executing tests th...
- Chunk 2: Establish SQA processes Coordination with relevant <b>software</b> processes....

### Google ADK | quiz
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are two major types of <b>software testing</b> - Black box testing : focuses on input, out...
- Chunk 2: ○ Look-At: Software requirement document ○ Look-For: the requirements are identified completely ○ Look-At: <b>software t...

### Google ADK | quiz
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are two major types of <b>software testing</b> - Black box testing : focuses on input, out...
- Chunk 2: ○ Look-At: Software requirement document ○ Look-For: the requirements are identified completely ○ Look-At: <b>software t...

### Google ADK | quiz
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are two major types of <b>software testing</b> - Black box testing : focuses on input, out...
- Chunk 2: ○ Look-At: Software requirement document ○ Look-For: the requirements are identified completely ○ Look-At: <b>software t...

### LangGraph | email
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are two major types of <b>software testing</b> - <b>Black box testing</b> : focuses on inp...
- Chunk 2: Establish SQA processes Coordination with relevant <b>software</b> processes....

### LangGraph | email
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are two major types of <b>software testing</b> - <b>Black box testing</b> : focuses on inp...
- Chunk 2: Establish SQA processes Coordination with relevant <b>software</b> processes....

### LangGraph | email
- Contexts retrieved: 2
- Chunk 1: <b>Software Testing</b> There are two major types of <b>software testing</b> - <b>Black box testing</b> : focuses on inp...
- Chunk 2: Establish SQA processes Coordination with relevant <b>software</b> processes....

### Google ADK | email
- Contexts retrieved: 0

### Google ADK | email
- Contexts retrieved: 0

### Google ADK | email
- Contexts retrieved: 0

