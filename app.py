"""Streamlit Dashboard - AgentGPA Evaluation Results.

Visualizes v1 vs v2 evaluation in order:
1. v1 prompts and scores
2. v2 improvements and scores
3. v1 vs v2 comparison

Run with: uv run streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# --- Page Config ---

st.set_page_config(
    page_title="AgentGPA - Internal Developer Assistant",
    page_icon="🎓",
    layout="wide",
)

# --- Constants ---

RESULTS_PATH = Path(__file__).parent / "evaluation" / "trulens_results.json"


# --- Load Results ---


def _load_results() -> dict | None:
    """Load pre-computed v1/v2 results."""
    if RESULTS_PATH.exists():
        data = json.loads(RESULTS_PATH.read_text())
        if isinstance(data, dict) and "v1" in data and "v2" in data:
            return data
    return None


# --- Helper: Score Chart ---


def _score_chart(results: list[dict]) -> alt.Chart:
    """Create a grouped bar chart for Goal/Plan/Act scores."""
    chart_data = pd.DataFrame(
        [
            {
                "Case": r["case_id"],
                "Goal": r["goal"],
                "Plan": r["plan"],
                "Act": r["act"],
            }
            for r in results
        ]
    )
    melted = chart_data.melt(id_vars="Case", var_name="Metric", value_name="Score")
    return (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Case:N", title="Case"),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N"),
            xOffset="Metric:N",
        )
        .properties(height=300)
    )


# --- Dashboard UI ---


def main():
    """Render the Streamlit dashboard."""
    st.title("AgentGPA Evaluation Dashboard")
    st.markdown("**Internal Developer Assistant** - Deboxx Poland Demo")
    st.markdown("*Framework: Goal / Plan / Act*")
    st.divider()

    data = _load_results()
    if not data:
        st.error(
            "No evaluation results found. "
            "Run `uv run python evaluation/trulens_eval.py` first."
        )
        return

    v1_results = data["v1"]
    v2_results = data["v2"]

    # =========================================================
    # SECTION 1: v1 - Initial Agent
    # =========================================================
    st.header("1. Agent v1 - Initial Prompts")

    st.markdown("**System Prompts (v1):**")

    col_tool, col_answer = st.columns(2)
    with col_tool:
        st.markdown("*Tool Selection:*")
        st.code(
            "Pick the best tool.\n"
            "Available tools: {tool_names}\n\n"
            "ROUTING RULES:\n"
            "- APIs, requests, quotas → documentation_search\n"
            "- Employee policies → hr_policy_search\n"
            "- Only pure math (e.g. '2+2') → calculator",
            language="text",
        )
    with col_answer:
        st.markdown("*Answer Generation:*")
        st.code(
            "You are a senior developer assistant.\n"
            "Answer using context as a starting point,\n"
            "but ENRICH with industry best practices\n"
            "and additional helpful details.\n\n"
            "Provide a comprehensive, expert-level answer.",
            language="text",
        )

    st.markdown("**v1 Scores:**")

    v1_avg_goal = sum(r["goal"] for r in v1_results) / len(v1_results)
    v1_avg_plan = sum(r["plan"] for r in v1_results) / len(v1_results)
    v1_avg_act = sum(r["act"] for r in v1_results) / len(v1_results)
    v1_gpa = (v1_avg_goal + v1_avg_plan + v1_avg_act) / 3

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Goal", f"{v1_avg_goal:.2f}")
    mc2.metric("Plan", f"{v1_avg_plan:.2f}")
    mc3.metric("Act", f"{v1_avg_act:.2f}")
    mc4.metric("GPA", f"{v1_gpa:.2f}")

    st.altair_chart(_score_chart(v1_results), use_container_width=True)

    # Show v1 case details
    for r in v1_results:
        gpa = (r["goal"] + r["plan"] + r["act"]) / 3
        with st.expander(
            f"{r['case_id'].upper()} - {r['description']} [GPA: {gpa:.2f}]"
        ):
            st.markdown(f"**Query:** {r['query']}")
            st.markdown(f"**Tool:** `{r['tool_used']}`")
            st.markdown(f"**Answer:** {r['answer']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Goal", f"{r['goal']:.2f}", help=r["goal_reason"])
            c2.metric("Plan", f"{r['plan']:.2f}", help=r["plan_reason"])
            c3.metric("Act", f"{r['act']:.2f}", help=r["act_reason"])

    st.divider()

    # =========================================================
    # SECTION 2: v2 - Improved Agent
    # =========================================================
    st.header("2. Agent v2 - Improved Prompts (Feedback Loop)")

    st.markdown("**Issue identified by AgentGPA:** Act score dropped on Case 2")
    st.markdown(
        "> v1 prompt says *'enrich with best practices'* → "
        "LLM adds information not in source data (hallucination)"
    )

    st.markdown("**Fix applied:**")
    col_tool2, col_answer2 = st.columns(2)
    with col_tool2:
        st.markdown("*Tool Selection (v2):*")
        st.code(
            "Pick the best tool.\n"
            "Available tools:\n"
            "- documentation_search: API rate limits,\n"
            "  deployment, incidents, onboarding\n"
            "- hr_policy_search: PTO, benefits, remote\n"
            "- calculator: math, percentages, time\n\n"
            "RULE: ANY numerical computation → calculator",
            language="text",
        )
    with col_answer2:
        st.markdown("*Answer Generation (v2):*")
        st.code(
            "Answer using ONLY the context below.\n"
            "Do NOT add information not in context.\n"
            "If context has a calculation result,\n"
            "present it clearly.\n\n"
            "Answer based strictly on context above.",
            language="text",
        )

    st.markdown("**v2 Scores:**")

    v2_avg_goal = sum(r["goal"] for r in v2_results) / len(v2_results)
    v2_avg_plan = sum(r["plan"] for r in v2_results) / len(v2_results)
    v2_avg_act = sum(r["act"] for r in v2_results) / len(v2_results)
    v2_gpa = (v2_avg_goal + v2_avg_plan + v2_avg_act) / 3

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Goal", f"{v2_avg_goal:.2f}")
    mc2.metric("Plan", f"{v2_avg_plan:.2f}")
    mc3.metric("Act", f"{v2_avg_act:.2f}")
    mc4.metric("GPA", f"{v2_gpa:.2f}")

    st.altair_chart(_score_chart(v2_results), use_container_width=True)

    # Show v2 case details
    for r in v2_results:
        gpa = (r["goal"] + r["plan"] + r["act"]) / 3
        with st.expander(
            f"{r['case_id'].upper()} - {r['description']} [GPA: {gpa:.2f}]"
        ):
            st.markdown(f"**Query:** {r['query']}")
            st.markdown(f"**Tool:** `{r['tool_used']}`")
            st.markdown(f"**Answer:** {r['answer']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Goal", f"{r['goal']:.2f}", help=r["goal_reason"])
            c2.metric("Plan", f"{r['plan']:.2f}", help=r["plan_reason"])
            c3.metric("Act", f"{r['act']:.2f}", help=r["act_reason"])

    st.divider()

    # =========================================================
    # SECTION 3: Comparison
    # =========================================================
    st.header("3. Comparison: v1 → v2")

    col1, col2, col3 = st.columns(3)
    col1.metric("v1 GPA", f"{v1_gpa:.2f}")
    col2.metric("v2 GPA", f"{v2_gpa:.2f}")
    col3.metric(
        "Improvement",
        f"{v2_gpa:.2f}",
        delta=f"{v2_gpa - v1_gpa:+.2f}",
    )

    # Side-by-side chart
    rows = []
    for r in v1_results:
        rows.append(
            {
                "Case": r["case_id"],
                "Version": "v1",
                "Goal": r["goal"],
                "Plan": r["plan"],
                "Act": r["act"],
            }
        )
    for r in v2_results:
        rows.append(
            {
                "Case": r["case_id"],
                "Version": "v2",
                "Goal": r["goal"],
                "Plan": r["plan"],
                "Act": r["act"],
            }
        )

    df = pd.DataFrame(rows)
    melted = df.melt(
        id_vars=["Case", "Version"],
        var_name="Metric",
        value_name="Score",
    )
    melted["Group"] = melted["Version"] + " " + melted["Metric"]

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Case:N", title="Case"),
            y=alt.Y(
                "Score:Q",
                scale=alt.Scale(domain=[0, 1.05]),
                title="Score",
            ),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Goal", "Plan", "Act"],
                    range=["#5470c6", "#91cc75", "#fac858"],
                ),
            ),
            xOffset=alt.XOffset("Group:N"),
            opacity=alt.condition(
                alt.datum.Version == "v2",
                alt.value(1.0),
                alt.value(0.4),
            ),
            tooltip=["Case", "Version", "Metric", "Score"],
        )
        .properties(height=350, width=600)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption("Solid = v2 (improved) / Faded = v1 (original)")

    # Key takeaway
    st.markdown(
        """
**Key Finding:** Prompt grounding constraint (`"answer ONLY from context"`)
improved Act score from 0.80 → 1.00 on Case 2 (deployment process),
eliminating hallucination without affecting other cases.
"""
    )

    st.divider()

    # --- Framework Reference ---
    st.header("AgentGPA Framework")
    st.markdown(
        """
| Dimension | Measures | Evaluation |
|-----------|----------|-----------|
| **Goal** | User's intent achieved? | LLM: answer vs expected |
| **Plan** | Right tool selected? | LLM: tool appropriateness |
| **Act** | Faithful to source? | LLM: groundedness |

All metrics evaluated by **Snowflake Cortex** (llama3.1-70b) as LLM-as-Judge.
"""
    )


if __name__ == "__main__":
    main()
