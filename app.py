"""Streamlit Dashboard - AgentGPA Evaluation Results.

Visualizes v1 vs v2 evaluation in order:
1. v1 prompts and scores
2. v2 improvements and scores
3. v1 vs v2 comparison

Run with: uv run streamlit run app.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

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
    st.markdown("**Internal Developer Assistant**")
    st.markdown("*AI Agent for developer questions: docs, policies, calculations*")
    st.divider()

    # =========================================================
    # Agent Architecture Diagram
    # =========================================================
    st.header("Agent Architecture")

    mermaid_diagram = """\
flowchart LR
    User([Developer Query])
    LLM_Router["LLM Router\\n(Tool Selection Prompt)"]
    DocSearch["documentation_search\\nInternal Docs"]
    HRSearch["hr_policy_search\\nHR Policies"]
    Calc["calculator\\nMath Engine"]
    LLM_Answer["LLM Answer Generator\\n(Answer Generation Prompt)"]
    Response([Agent Response])

    User --> LLM_Router
    LLM_Router -->|technical| DocSearch
    LLM_Router -->|policy| HRSearch
    LLM_Router -->|math| Calc
    DocSearch --> LLM_Answer
    HRSearch --> LLM_Answer
    Calc --> Response
    LLM_Answer --> Response
"""

    st.components.v1.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <div class="mermaid">
        {mermaid_diagram}
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    primaryColor: '#5470c6',
                    primaryTextColor: '#fff',
                    lineColor: '#91cc75',
                    secondaryColor: '#fac858'
                }}
            }});
        </script>
        """,
        height=250,
    )

    st.caption(
        "LLM Router and Answer Generator are powered by Snowflake Cortex "
        "(llama3.1-70b). Calculator results bypass the Answer Generator."
    )
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
            st.markdown(f"**Expected:** {r.get('expected_answer', 'N/A')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Goal", f"{r['goal']:.2f}", help=r["goal_reason"])
            c2.metric("Plan", f"{r['plan']:.2f}", help=r["plan_reason"])
            c3.metric("Act", f"{r['act']:.2f}", help=r["act_reason"])

    st.divider()

    # =========================================================
    # SECTION 2: v2 - Improved Agent
    # =========================================================
    st.header("2. Agent v2 - Feedback Loop Improvements")

    st.markdown("**Issues identified by AgentGPA on Case 2:**")
    st.markdown(
        "1. Knowledge base missing internal coding standards "
        "→ LLM cannot find correct answer\n"
        "2. Prompt says *'enrich with best practices'* "
        "→ LLM fabricates PEP8 answer (4 spaces instead of our 2 spaces)"
    )

    st.markdown("**Fixes applied:**")

    # Fix 1: Knowledge base
    st.markdown("*Fix 1 - Knowledge base expansion:*")
    st.markdown(
        """
<div style="font-family: monospace; font-size: 0.82em; line-height: 1.6; background: #1e1e1e; padding: 12px; border-radius: 6px;">
<span style="color: #3fb950;">+ doc-006: "Python Coding Standards"</span><br>
<span style="color: #3fb950;">+ "All Python code in our monorepo uses</span><br>
<span style="color: #3fb950;">+  2-space indentation. This differs from</span><br>
<span style="color: #3fb950;">+  PEP8 (4 spaces) and is enforced by our</span><br>
<span style="color: #3fb950;">+  pre-commit hooks."</span><br>
</div>
""",
        unsafe_allow_html=True,
    )

    # Fix 2: Prompt diff
    st.markdown("*Fix 2 - Answer generation prompt (grounding constraint):*")
    st.markdown(
        """
<div style="font-family: monospace; font-size: 0.82em; line-height: 1.6; background: #1e1e1e; padding: 12px; border-radius: 6px;">
<span style="color: #f85149;">- You are a senior developer assistant with</span><br>
<span style="color: #f85149;">-   expertise in deployment, CI/CD...</span><br>
<span style="color: #f85149;">- Answer using context as a starting point,</span><br>
<span style="color: #f85149;">-   but ENRICH with industry best practices</span><br>
<span style="color: #f85149;">-   and additional helpful details.</span><br>
<span style="color: #f85149;">- Provide comprehensive, expert-level answer.</span><br>
<br>
<span style="color: #3fb950;">+ Answer using ONLY the context below.</span><br>
<span style="color: #3fb950;">+ Do NOT add information not in context.</span><br>
<span style="color: #3fb950;">+ If not enough info, say</span><br>
<span style="color: #3fb950;">+   "I don't have enough information".</span><br>
<span style="color: #3fb950;">+ Answer based strictly on context above.</span><br>
</div>
""",
        unsafe_allow_html=True,
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
            st.markdown(f"**Expected:** {r.get('expected_answer', 'N/A')}")
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
**Key Finding:** Case 2 (Python indentation rule) — v1 hallucinates PEP8
(4 spaces) because the doc is missing AND the prompt says "enrich."
v2 fixes both: adds internal doc (2 spaces) + grounding constraint.
"""
    )

    st.divider()

    # =========================================================
    # SECTION 4: Observability
    # =========================================================
    st.header("4. Observability — Agent Trace")
    st.markdown(
        "Step-by-step execution trace showing which tool was selected, "
        "what was retrieved, and how the answer was generated."
    )

    for r in v2_results:
        gpa = (r["goal"] + r["plan"] + r["act"]) / 3
        timings = r.get("step_timings", {})
        token_usage = r.get("token_usage", {})
        prompts = r.get("prompts", {})

        with st.expander(
            f"{r['case_id'].upper()} — {r['description']} [GPA: {gpa:.2f}]",
            expanded=(r["case_id"] == "case_3"),
        ):
            # Step 1: Tool Selection
            t1 = timings.get("tool_selection_s", "")
            t1_label = f" ⏱ {t1}s" if t1 != "" else ""
            st.markdown(f"**Step 1 — Tool Selection** `[TOOL span]`{t1_label}")
            col_q, col_t = st.columns([3, 1])
            col_q.markdown(f"*Query:* {r['query']}")
            col_t.success(f"`{r['tool_used']}`")
            sel_usage = token_usage.get("tool_selection", {})
            if sel_usage.get("total_tokens"):
                st.caption(
                    f"Tokens — prompt: {sel_usage['prompt_tokens']} / "
                    f"completion: {sel_usage['completion_tokens']} / "
                    f"total: {sel_usage['total_tokens']}"
                )
            if r.get("selection_raw_response"):
                with st.expander("LLM raw response (tool selection)"):
                    st.text(r["selection_raw_response"])
            if prompts.get("tool_selection"):
                with st.expander("Prompt sent to LLM (tool selection)"):
                    st.code(
                        prompts["tool_selection"].replace(
                            "{tool_names}", r["tool_used"]
                        ),
                        language="text",
                    )

            # Step 2: Tool Execution
            t2 = timings.get("tool_execution_s", "")
            t2_label = f" ⏱ {t2}s" if t2 != "" else ""
            st.markdown(f"**Step 2 — Tool Execution** `[RETRIEVAL span]`{t2_label}")
            context = r.get("retrieved_context", "")
            if context:
                st.code(context[:400], language="text")
            else:
                st.info("Calculator: deterministic result, no document retrieval.")

            # Step 3: Answer Generation
            t3 = timings.get("answer_generation_s", "")
            t3_label = f" ⏱ {t3}s" if t3 != "" else ""
            st.markdown(
                f"**Step 3 — Answer Generation** `[AGENT span]`{t3_label}"
            )
            st.info(r["answer"])
            ans_usage = token_usage.get("answer_generation", {})
            if ans_usage.get("total_tokens"):
                st.caption(
                    f"Tokens — prompt: {ans_usage['prompt_tokens']} / "
                    f"completion: {ans_usage['completion_tokens']} / "
                    f"total: {ans_usage['total_tokens']}"
                )
            if prompts.get("answer_generation"):
                with st.expander("Prompt sent to LLM (answer generation)"):
                    st.code(prompts["answer_generation"], language="text")

            # GPA scores
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Goal", f"{r['goal']:.2f}", help=r["goal_reason"])
            c2.metric("Plan", f"{r['plan']:.2f}", help=r["plan_reason"])
            c3.metric("Act", f"{r['act']:.2f}", help=r["act_reason"])
            if timings.get("total_s"):
                c4.metric("Total time", f"{timings['total_s']}s")

    # --- Live Query ---
    st.subheader("Try a Query")
    st.markdown("Run the agent live and inspect the trace in real time.")

    col_ver, col_q = st.columns([1, 3])
    version_select = col_ver.radio(
        "Agent version", ["v1", "v2"], index=1, horizontal=True
    )
    user_query = col_q.text_input(
        "Query",
        placeholder="e.g. What is the Python indentation rule in our codebase?",
    )

    if st.button("Run Agent", type="primary") and user_query.strip():
        from src.agent import InternalDeveloperAssistant

        with st.spinner("Running agent..."):
            agent = InternalDeveloperAssistant(version=version_select)
            response = agent.run(user_query)

        st.markdown("**Trace:**")
        timings = response.step_timings
        token_usage = response.token_usage
        prompts = response.prompts

        # Step 1
        t1 = timings.get("tool_selection_s", "")
        st.markdown(
            f"**Step 1 — Tool Selection** `[TOOL span]`"
            + (f" ⏱ {t1}s" if t1 != "" else "")
        )
        col_lq, col_lt = st.columns([3, 1])
        col_lq.markdown(f"*Query:* {user_query}")
        col_lt.success(f"`{response.tool_used}`")
        sel_usage = token_usage.get("tool_selection", {})
        if sel_usage.get("total_tokens"):
            st.caption(
                f"Tokens — prompt: {sel_usage['prompt_tokens']} / "
                f"completion: {sel_usage['completion_tokens']} / "
                f"total: {sel_usage['total_tokens']}"
            )
        if response.selection_raw_response:
            with st.expander("LLM raw response (tool selection)"):
                st.text(response.selection_raw_response)
        if prompts.get("tool_selection"):
            with st.expander("Prompt sent to LLM (tool selection)"):
                st.code(prompts["tool_selection"], language="text")

        # Step 2
        t2 = timings.get("tool_execution_s", "")
        st.markdown(
            f"**Step 2 — Tool Execution** `[RETRIEVAL span]`"
            + (f" ⏱ {t2}s" if t2 != "" else "")
        )
        tool_results = response.tool_output.get("results", [])
        if tool_results and "message" not in tool_results[0]:
            live_context = tool_results[0].get("content", "")
            st.code(live_context[:400], language="text")
        elif response.tool_used == "calculator":
            calc_result = response.tool_output.get("result")
            st.info(f"Calculator result: {calc_result}")
        else:
            st.warning("No context retrieved.")

        # Step 3
        t3 = timings.get("answer_generation_s", "")
        st.markdown(
            f"**Step 3 — Answer Generation** `[AGENT span]`"
            + (f" ⏱ {t3}s" if t3 != "" else "")
        )
        st.info(response.answer)
        ans_usage = token_usage.get("answer_generation", {})
        if ans_usage.get("total_tokens"):
            st.caption(
                f"Tokens — prompt: {ans_usage['prompt_tokens']} / "
                f"completion: {ans_usage['completion_tokens']} / "
                f"total: {ans_usage['total_tokens']}"
            )
        if prompts.get("answer_generation"):
            with st.expander("Prompt sent to LLM (answer generation)"):
                st.code(prompts["answer_generation"], language="text")

        if timings.get("total_s"):
            st.metric("Total execution time", f"{timings['total_s']}s")

    st.divider()

    # --- Framework Reference ---
    st.header("AgentGPA Framework")
    st.markdown(
        """
| Dimension | Measures | Evaluation | Formal Metric |
|-----------|----------|-----------|---------------|
| **Goal** | User's intent achieved? | LLM: answer vs expected | 1A Answer Correctness, 1B Answer Relevance |
| **Plan** | Right tool selected? | LLM: tool appropriateness | 4B Tool Selection |
| **Act** | Faithful to source? | LLM: groundedness | 1C Groundedness |

All metrics evaluated by **Snowflake Cortex** (llama3.1-70b) as LLM-as-Judge.
"""
    )


if __name__ == "__main__":
    main()
