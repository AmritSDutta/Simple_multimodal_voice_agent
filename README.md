# FlowCheck

# Using Orchestrator–Worker Architecture

## Overview

FlowCheck implements a parallel evaluation pipeline for decision automation using **LangGraph** with an **orchestrator–worker fan‑out/fan‑in pattern**. The system:

1. Extracts and summarizes input context
2. Dispatches parallel decision evaluators
3. Aggregates all outputs into a unified final report

This design enables scalable execution while ensuring deterministic aggregation of results.

---

## Architectural Components

### ✅ Summarizer (Global Context Builder)

* Node: `summarizer`
* Model: **gemini-2.5-flash-lite**
* Responsibilities:

  * interpret raw issue input
  * generate structured `issue`
  * produce `sub_issues_decision: list[DecisionOutput]`

### ✅ Fan‑Out Executor (Orchestrator)

* Implemented as **conditional edge**, not a node
* Function: `assign_workers`
* Generates:

  ```python
  [Send("subtask", payload) ...]
  ```
* Each dispatched branch receives isolated `sub_issue`

### ✅ Worker Nodes (Parallel Evaluation)

* Node: `subtask`
* Model: **gpt‑5‑nano**
* Execution via:

  ```python
  Runner.run(starting_agent, input, context)
  ```
* Returns:

  ```python
  {"completed_sub_issues_decision": [DecisionOutput]}
  ```

### ✅ Fan‑In Aggregator

* Node: `combiner`
* Input merged automatically because:

  ```python
  completed_sub_issues_decision: Annotated[list[DecisionOutput], operator.add]
  ```
* Produces:

  ```python
  final_report: CombinedPlan
  ```

---

## State Definition

```python
class State(TypedDict):
    retry_count: Annotated[int, add]
    messages: Annotated[list[BaseMessage], add_messages]
    issue: str
    sub_issues_decision: list[DecisionOutput]
    sub_issue: NotRequired[DecisionOutput]
    completed_sub_issues_decision: Annotated[list[DecisionOutput], operator.add]
    final_report: CombinedPlan
```

### Why this matters

* `operator.add` enables list concatenation during fan‑in
* `sub_issue` is optional because it only exists inside worker branches
* messages and retry_count remain compatible with LangGraph execution

---

## Execution Flow

```
START
  ↓
summarizer
  ↓
assign_workers  (conditional edge)
  ├─ Send → subtask (worker 1)
  ├─ Send → subtask (worker 2)
  ├─ Send → subtask (worker 3)
  …
  ↓ (after all workers complete)
combiner
  ↓
END
```

---

## Key Rules and Guarantees

✅ Fan‑out must return Send(), not dict
✅ Fan‑out must not be registered as a node
✅ Worker return values must be dicts
✅ Worker outputs must be lists
✅ Shared state must be passed into Send payload
✅ Dynamic instructions must escape braces if using f‑strings
✅ Null fields like `notes` must be normalized

---
## 🔍 Example: Dynamic Evaluator Prompt (run_connectivity_diagnostics)

Each evaluator receives a dynamically constructed prompt based on its `decision_id`.  
Below is an example of the **actual prompt** used for the `run_connectivity_diagnostics` evaluator.

### **Evaluator Prompt (Auto-Generated)**

```
as a agent of run_connectivity_diagnostics. 
You are an automated decision evaluator.

Input:
- decision_id: run_connectivity_diagnostics
- context: unstructured text containing events, logs, symptoms, actions, or user reports.

Task:
1. Read and interpret the context.
2. Based solely on the meaning of the decision_id, determine if action is required:
   - Triggered by network failures, unreachable services, or packet loss indications.
3. Return:
   - decision: true if action is warranted, false otherwise
   - confidence: 0.0–1.0 expressing certainty
   - model: name of the model producing the output
   - notes: concise reasoning (optional) , must be maximum 10 words.
   - latency_ms: leave empty

Output JSON strictly in the following structure:

{
  "decision_id": "run_connectivity_diagnostics",
  "decision": true or false,
  "confidence": 0.0,
  "model": "gpt-5-nano",
  "notes": "short rationale",
  "latency_ms": null
}
optimize output token usage without compromising on quality of output.
Help them with their questions.
```

---

### **Sample Output**

```json
{
  "decision_id": "run_connectivity_diagnostics",
  "decision": true,
  "confidence": 0.81,
  "model": "gpt-5-nano",
  "notes": "Latency spikes and packet loss reported",
  "latency_ms": null
}
```
### Notes

- Every evaluator follows the same structure; only decision_id, decision logic description, and sample schema differ.
- Prompts remain short to minimize cost while preserving clarity.
---

The executor uses these outputs to assemble the final CombinedPlan.
---

## Model Selection Rationale

| Component       | Model                  | Reason                               |
| --------------- | ---------------------- | ------------------------------------ |
| summarizer      | **gpt‑2.5‑flash‑lite** | inexpensive global contextualization |
| subtask workers | **gpt‑5‑nano**         | fast, structured, parallelizable     |
| final combiner  | inherits context       | purely deterministic merging         |

This yields cost‑efficient scaling because heavy reasoning doesn’t run per worker.

---

## Suggested Enhancements

✅ concurrency limiter (e.g., max 3 workers)
✅ telemetry: latency, decision rate, disagreement counts
✅ cost attribution per decision_id
✅ retry policy only at worker level

---

## When to Use This Architecture

Use it if you need:
✅ independent evaluations per decision type
✅ consistent aggregation
✅ heterogeneous model assignment
✅ parallelism with deterministic merge semantics

Do **not** use if:
❌ decisions depend on each other
❌ ordering impacts evaluation

---
# FlowCheck UI (Streamlit Client)

A lightweight UI for interacting with the FlowCheck LangGraph deployment over REST. It supports large incident input, run execution, polling run status, and displaying the final `final_report` from thread state.

## Requirements
```bash
pip install streamlit requests
```

## Configuration
Edit at top of `app.py`:
```python
DEPLOYMENT_URL = "http://localhost:2024"
ASSISTANT_ID = "agent"
```

## Run
```bash
python -m streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

## Features
- Large text issue input
- Sends request to LangGraph deployment
- Shows compact “running” status
- Fetches thread state after success
- Displays formatted `final_report`

## Troubleshooting
| Issue | Fix |
|-------|-----|
| 422 on thread create | Add `json={}` body |
| No final report | Read from thread, not run |
| Connection failure | Check deployment URL & server |

## Optional Enhancements
- Show node transitions
- Export report file
- Use new thread per run
- Add auth headers




## License

Internal architectural documentation for FlowCheck.
