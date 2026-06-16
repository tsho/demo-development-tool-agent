"""Streamlit Dashboard - AgentGPA Evaluation Results.

Visualizes the evaluation results of the Internal Developer Assistant.
Run with: uv run streamlit run app.py
"""

import streamlit as st

from src.agent import InternalDeveloperAssistant

# --- Page Config ---

st.set_page_config(
    page_title="AgentGPA - Internal Developer Assistant",
    page_icon="🎓",
    layout="wide",
)

# --- Test Cases ---

TEST_CASES = [
    {
        "id": "case_1",
        "query": "How many PTO days do employees receive?",
        "mode": "correct",
        "expected_tool": "hr_policy_search",
        "description": "Correct tool + faithful answer",
    },
    {
        "id": "case_2",
        "query": "How many vacation days do employees receive?",
        "mode": "broken",
        "expected_tool": "hr_policy_search",
        "description": "Hallucination (Faithfulness drops)",
    },
    {
        "id": "case_3",
        "query": "What is 15% of $1200?",
        "mode": "correct",
        "expected_tool": "calculator",
        "description": "Correct tool (Calculator) + correct result",
    },
    {
        "id": "case_4",
        "query": "What is 15% of $1200?",
        "mode": "broken",
        "expected_tool": "calculator",
        "description": "Wrong tool (HR Search instead of Calculator)",
    },
]


# --- Evaluation Logic ---


def _evaluate_faithfulness(response, test_case):
    """Evaluate faithfulness of the response."""
    import re

    tool_output = response.tool_output
    answer = response.answer.lower()

    if response.tool_used == "calculator":
        result = tool_output.get("result")
        if result is not None and str(int(result)) in answer:
            return 1.0, "Answer matches calculator output"
        if result is not None:
            return 0.5, "Calculator produced result but answer doesn't reflect it"
        return 0.0, "Calculator failed and answer is not grounded"

    results = tool_output.get("results", [])
    if not results or "message" in results[0]:
        if "couldn't find" in answer:
            return 1.0, "Correctly reported no results found"
        return 0.0, "No source data but agent generated answer (HALLUCINATION)"

    source_content = results[0].get("content", "").lower()
    numbers_in_answer = re.findall(r"\d+", answer)
    numbers_in_source = re.findall(r"\d+", source_content)

    if numbers_in_answer:
        grounded = [n for n in numbers_in_answer if n in numbers_in_source]
        if len(grounded) == len(numbers_in_answer):
            return 1.0, "All claims grounded in source data"
        if len(grounded) > 0:
            ratio = len(grounded) / len(numbers_in_answer)
            return ratio, f"{len(grounded)}/{len(numbers_in_answer)} claims grounded"
        return 0.0, "Facts not present in source data (HALLUCINATION)"

    return 0.8, "Generally grounded (no numeric claims to verify)"


def _evaluate_tool_selection(response, test_case):
    """Evaluate tool selection accuracy."""
    expected = test_case["expected_tool"]
    actual = response.tool_used
    if actual == expected:
        return 1.0, f"Correctly selected '{actual}'"
    return 0.0, f"Selected '{actual}' but expected '{expected}'"


def _evaluate_relevancy(response, test_case):
    """Evaluate answer relevancy."""
    expected_content = test_case.get("expected_answer_contains", "").lower()
    answer = response.answer.lower()
    query_lower = test_case["query"].lower()

    # For cases without expected_answer_contains, use tool match as proxy
    if not expected_content:
        if response.tool_used == test_case["expected_tool"]:
            return 1.0, "Answer from correct tool is relevant"
        return 0.0, "Answer from wrong tool is irrelevant"

    has_expected = expected_content in answer

    if "%" in query_lower or "calculate" in query_lower:
        is_topical = any(w in answer for w in ["answer", "result", "equal", "$", "is"])
    elif "pto" in query_lower or "vacation" in query_lower:
        is_topical = any(w in answer for w in ["days", "pto", "policy", "vacation"])
    else:
        is_topical = True

    if has_expected and is_topical:
        return 1.0, "Directly addresses question with correct info"
    if is_topical and not has_expected:
        return 0.5, "Topically relevant but incorrect info"
    if has_expected and not is_topical:
        return 0.5, "Contains expected info but off-topic"
    return 0.0, "Not relevant to the question"


@st.cache_data
def run_all_cases():
    """Run all test cases and return results."""
    results = []
    for case in TEST_CASES:
        agent = InternalDeveloperAssistant(mode=case["mode"])
        response = agent.run(case["query"])

        f_score, f_reason = _evaluate_faithfulness(response, case)
        t_score, t_reason = _evaluate_tool_selection(response, case)
        r_score, r_reason = _evaluate_relevancy(response, case)

        results.append(
            {
                "case_id": case["id"],
                "query": case["query"],
                "mode": case["mode"],
                "description": case["description"],
                "tool_used": response.tool_used,
                "expected_tool": case["expected_tool"],
                "answer": response.answer,
                "faithfulness": f_score,
                "faithfulness_reason": f_reason,
                "tool_selection": t_score,
                "tool_selection_reason": t_reason,
                "relevancy": r_score,
                "relevancy_reason": r_reason,
            }
        )
    return results


# --- Dashboard UI ---


def main():
    """Render the Streamlit dashboard."""
    st.title("AgentGPA Evaluation Dashboard")
    st.markdown("**Internal Developer Assistant** - Deboxx Poland Demo")
    st.divider()

    results = run_all_cases()

    # --- Summary Metrics ---
    st.header("Overall Scores")

    avg_f = sum(r["faithfulness"] for r in results) / len(results)
    avg_t = sum(r["tool_selection"] for r in results) / len(results)
    avg_r = sum(r["relevancy"] for r in results) / len(results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Faithfulness", f"{avg_f:.0%}")
    col2.metric("Tool Selection", f"{avg_t:.0%}")
    col3.metric("Relevancy", f"{avg_r:.0%}")
    col4.metric("Overall", f"{(avg_f + avg_t + avg_r) / 3:.0%}")

    st.divider()

    # --- Comparison Chart ---
    st.header("Per-Case Scores")

    import altair as alt
    import pandas as pd

    chart_data = pd.DataFrame(
        [
            {
                "Case": r["case_id"],
                "Faithfulness": r["faithfulness"],
                "Tool Selection": r["tool_selection"],
                "Relevancy": r["relevancy"],
            }
            for r in results
        ]
    )
    melted = chart_data.melt(id_vars="Case", var_name="Metric", value_name="Score")
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Case:N", title="Case"),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N"),
            xOffset="Metric:N",
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    st.divider()

    # --- Case Details ---
    st.header("Case Details")

    for r in results:
        is_pass = (r["faithfulness"] + r["tool_selection"] + r["relevancy"]) / 3 >= 0.8
        status_icon = "✅" if is_pass else "❌"
        mode_badge = "🟢 correct" if r["mode"] == "correct" else "🔴 broken"

        with st.expander(
            f"{status_icon} {r['case_id'].upper()} - {r['description']} ({mode_badge})"
        ):
            st.markdown(f"**Query:** {r['query']}")
            st.markdown(
                f"**Tool:** `{r['tool_used']}` (expected: `{r['expected_tool']}`)"
            )
            st.markdown(f"**Answer:** {r['answer']}")

            st.markdown("---")
            st.markdown("**Scores:**")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "Faithfulness",
                f"{r['faithfulness']:.1f}",
                help=r["faithfulness_reason"],
            )
            mc2.metric(
                "Tool Selection",
                f"{r['tool_selection']:.1f}",
                help=r["tool_selection_reason"],
            )
            mc3.metric(
                "Relevancy",
                f"{r['relevancy']:.1f}",
                help=r["relevancy_reason"],
            )

            st.caption(
                f"Faithfulness: {r['faithfulness_reason']} | "
                f"Tool: {r['tool_selection_reason']} | "
                f"Relevancy: {r['relevancy_reason']}"
            )

    st.divider()

    # --- Key Takeaways ---
    st.header("Key Demo Points")
    st.markdown(
        """
- **Case 1 vs 2**: Same tool, different output
  → **Faithfulness** metric catches hallucination
- **Case 3 vs 4**: Same query, different tool
  → **Tool Selection** metric catches wrong routing
- **All cases**: **Relevancy** evaluates answer-question alignment independently
"""
    )


if __name__ == "__main__":
    main()
