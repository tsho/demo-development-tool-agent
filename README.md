# Internal Developer Assistant - AgentGPA Demo

Deboxx Poland demo: AI Agent evaluation using the **Goal / Plan / Act** framework with Snowflake Cortex LLM-as-Judge.

## Demo Story

An LLM-based "Internal Developer Assistant" with 3 tools is evaluated in two versions:

- **v1** — Weak prompts + incomplete knowledge base → hallucination on Case 3
- **v2** — Grounded prompts + expanded docs → correct answers

| Case | Query | Expected | v1 Issue |
|------|-------|----------|----------|
| 1 | "What is 25% of 800?" | 200 | None (baseline) |
| 2 | "API handles 50 req/s, how many in 30 min?" | 90,000 | None (baseline) |
| 3 | "What is the Python indentation rule in our codebase?" | 2-space indent | Hallucination: PEP8 (4 spaces) |

### v1 → v2 Improvement (Feedback Loop)

**Problem:** v1 Case 3 gets Goal=0.0, Act=0.0 because:
1. Internal coding standards doc is missing from knowledge base
2. Prompt says "enrich with industry best practices" → LLM answers PEP8 (4 spaces)

**Fix (two axes):**
1. **Data**: Added `doc-006: Python Coding Standards` (2-space indent rule)
2. **Prompt**: Changed from "enrich with best practices" → "answer ONLY from context"

**Result:** Case 3 GPA improved from 0.33 → 0.83.

## AgentGPA Framework

| Dimension | Measures | Evaluation Method |
|-----------|----------|-------------------|
| **Goal** | User's intent achieved? | LLM judge: answer vs expected |
| **Plan** | Right tool selected? | LLM judge: tool appropriateness |
| **Act** | Faithful to source data? | LLM judge: groundedness |

All metrics scored 0.0–1.0 by **Snowflake Cortex** (`llama3.1-70b`) as LLM-as-Judge.

## Architecture

```
Developer Query → LLM Router (Tool Selection Prompt)
                     ├─ technical → documentation_search → LLM Answer Generator → Response
                     ├─ policy    → hr_policy_search    → LLM Answer Generator → Response
                     └─ math      → calculator          → Response (bypasses LLM)
```

- LLM Router and Answer Generator powered by Snowflake Cortex (`llama3.1-70b`)
- v1/v2 behavior controlled by different prompts (`src/prompts_v1.py`, `src/prompts_v2.py`)
- v1 uses `data/documentation.json`, v2 uses `data/documentation_v2.json` (with coding standards)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Snowflake account with Cortex access

## Setup

```bash
git clone <repo-url>
cd demo-development-tool-agent
git checkout feature/deboxx-poland-agent-demo

uv sync

cp .env.example .env
# Edit .env with your Snowflake credentials
```

## 1. Run the AI Agent

```bash
uv run python main.py
```

Programmatic usage:

```python
from src.agent import InternalDeveloperAssistant

agent = InternalDeveloperAssistant(version="v2", snowpark_session=session)
response = agent.run("What is the Python indentation rule in our codebase?")
# response.answer → "2-space indentation..."
# response.tool_used → "documentation_search"
```

## 2. Run Evaluation

Runs all 3 test cases for v1 and v2, scores with Cortex LLM-as-Judge:

```bash
uv run python evaluation/trulens_eval.py
```

Results saved to `evaluation/trulens_results.json`.

## 3. Launch Streamlit Dashboard

```bash
uv run streamlit run app.py
```

Opens at http://localhost:8501 and displays:
- **Agent Architecture** — Mermaid diagram of tool routing
- **Section 1** — v1 prompts + scores
- **Section 2** — v2 improvements (diff view) + scores
- **Section 3** — v1 vs v2 comparison chart (Goal/Plan/Act color-coded)

## Project Structure

```
├── src/
│   ├── agent.py           # LLM-based agent with Cortex tool selection + answer gen
│   ├── prompts_v1.py      # v1 prompts (weak: "enrich with best practices")
│   ├── prompts_v2.py      # v2 prompts (grounded: "ONLY from context")
│   └── tools.py           # Tool implementations (doc search, HR search, calculator)
├── data/
│   ├── documentation.json      # v1 docs (no coding standards)
│   ├── documentation_v2.json   # v2 docs (includes Python coding standards)
│   └── hr_policies.json        # HR policies corpus
├── evaluation/
│   ├── trulens_eval.py         # AgentGPA evaluation (Cortex LLM-as-Judge)
│   └── trulens_results.json    # Pre-computed evaluation results
├── app.py                 # Streamlit dashboard
├── main.py                # Quick interactive demo
├── .env.example           # Environment variable template
├── pyproject.toml         # Dependencies + ruff config (Google style)
└── .python-version        # Python 3.11
```

## Linting

```bash
uv run ruff check .
uv run ruff format .
```

## Key Takeaway

With 3 test cases and an internal developer assistant, AgentGPA evaluates agent quality on **Goal** (intent achieved), **Plan** (right tool), and **Act** (faithful to source). The v1→v2 feedback loop demonstrates how prompt engineering + knowledge base expansion eliminates hallucination.
