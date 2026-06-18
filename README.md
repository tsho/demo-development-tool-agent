# Internal Developer Assistant - AgentGPA Demo

Deboxx Poland demo: AI Agent evaluation with TruLens + AgentGPA metrics.

## Demo Story

An "Internal Developer Assistant" with 3 tools: Documentation Search, HR Policy Search, and Calculator. Four cases demonstrate how evaluation metrics catch different failure modes.

| Case | Query | Behavior | Metric Impact |
|------|-------|----------|---------------|
| 1 | "How many PTO days do employees receive?" | Correct tool + faithful answer | All metrics pass |
| 2 | "How many vacation days do employees receive?" | Hallucinated answer | **Faithfulness drops** |
| 3 | "What is 15% of $1200?" | Calculator -> correct result | All metrics pass |
| 4 | "What is 15% of $1200?" | HR Search (wrong tool) | **Tool Selection drops**, **Relevancy drops** |

## Metrics Evaluated

- **Groundedness / Faithfulness** - Is the answer grounded in retrieved data?
- **Answer Relevance** - Does the answer address the user's question?
- **Tool Selection Accuracy** - Did the agent use the correct tool?

## Prerequisites

- Python 3.11 (required for `trulens-providers-cortex`)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd demo-development-tool-agent
git checkout feature/deboxx-poland-agent-demo

# Install all dependencies
uv sync
```

## 1. Run the AI Agent

The agent can be run in two modes: `correct` (expected behavior) and `broken` (demonstrates failure modes). The quick demo runs all 4 cases sequentially:

```bash
uv run python main.py
```

This will output each case showing:
- The user query
- Which tool the agent selected
- The agent's answer

You can also use the agent programmatically:

```python
from src.agent import InternalDeveloperAssistant

# Correct mode
agent = InternalDeveloperAssistant(mode="correct")
response = agent.run("How many PTO days do employees receive?")
print(response.answer)

# Broken mode (demonstrates failures)
agent_broken = InternalDeveloperAssistant(mode="broken")
response = agent_broken.run("How many vacation days do employees receive?")
print(response.answer)  # Hallucinated answer
```

## 2. Run Evaluation

### Rule-based Evaluation (no LLM required)

Runs deterministic scoring for Faithfulness, Tool Selection, and Relevancy:

```bash
uv run python evaluation/run_eval.py
```

Output includes per-case scores and a summary table showing which cases pass/fail.

### TruLens Evaluation (LLM-as-Judge)

Uses an LLM to judge Groundedness and Answer Relevance via TruLens.

#### With Snowflake Cortex (default provider)

```bash
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_user"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"

uv run python evaluation/trulens_eval.py
```

#### With OpenAI (fallback)

```bash
export OPENAI_API_KEY="sk-..."

uv run python evaluation/trulens_eval.py
```

Results are stored in `evaluation/trulens_eval.sqlite`. You can also launch the TruLens built-in dashboard:

```bash
uv run trulens-dashboard
```

## 3. Launch Streamlit Dashboard

The Streamlit app visualizes evaluation results with an interactive UI. No LLM or API keys required — it runs the rule-based evaluation internally.

```bash
uv run streamlit run app.py
```

Opens at http://localhost:8501 and displays:
- **Overall Scores** — Average Faithfulness, Tool Selection, Relevancy metrics
- **Per-Case Scores** — Grouped bar chart comparing all metrics side-by-side per case
- **Case Details** — Expandable sections with query, tool used, answer, and score explanations

## Project Structure

```
├── src/
│   ├── agent.py          # Agent with @instrument decorators + correct/broken modes
│   └── tools.py          # Tool implementations (Doc Search, HR Search, Calculator)
├── data/
│   ├── documentation.json     # Internal docs corpus
│   └── hr_policies.json       # HR policies corpus (ground truth: 20 PTO days)
├── evaluation/
│   ├── run_eval.py       # Rule-based evaluation (no LLM needed)
│   └── trulens_eval.py   # TruLens LLM-as-Judge evaluation (Cortex/OpenAI)
├── app.py                # Streamlit dashboard for evaluation visualization
├── main.py               # Quick interactive demo
├── pyproject.toml         # Dependencies + ruff config (Google style)
└── .python-version        # Python 3.11 (required for trulens-providers-cortex)
```

## Linting

```bash
uvx ruff check .
uvx ruff format .
```

## Key Takeaway

With just 4 test cases and an "internal assistant" scenario, AgentGPA evaluates all three critical dimensions of agent quality: **Faithfulness**, **Tool Selection**, and **Relevancy**.
