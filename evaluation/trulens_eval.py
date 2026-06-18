"""AgentGPA Evaluation - Deboxx Poland Demo.

Evaluates the Internal Developer Assistant v1 and v2 using Cortex LLM-as-Judge.
Compares Goal / Plan / Act scores before and after prompt improvement.

Results are saved to evaluation/trulens_results.json for the Streamlit dashboard.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from snowflake.snowpark import Session  # noqa: E402

from src.agent import InternalDeveloperAssistant  # noqa: E402

# --- Constants ---

RESULTS_PATH = Path(__file__).parent / "trulens_results.json"


# --- Snowflake Connection ---


def _create_snowpark_session() -> Session:
    """Create a Snowflake Snowpark session from environment variables."""
    required = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_USER_PASSWORD",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    return Session.builder.configs(
        {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "role": os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        }
    ).create()


# --- LLM-as-Judge Functions ---


def _judge_goal(
    session, query: str, answer: str, expected_answer: str
) -> tuple[float, str]:
    """Judge whether the agent achieved the user's goal."""
    from snowflake.cortex import complete

    prompt = (
        "You are an evaluation judge. Rate whether the AGENT ANSWER "
        "achieves the USER'S GOAL on a scale from 0.0 to 1.0.\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"EXPECTED CORRECT ANSWER:\n{expected_answer}\n\n"
        f"AGENT ANSWER:\n{answer}\n\n"
        "Scoring:\n"
        "- 1.0: Fully achieves the goal with correct information.\n"
        "- 0.7-0.9: Mostly achieves with minor inaccuracies.\n"
        "- 0.3-0.6: Partially achieves but significant errors.\n"
        "- 0.0-0.2: Completely fails to achieve the goal.\n\n"
        'Respond ONLY as JSON: {"score": <float>, "reason": "<brief>"}'
    )

    resp = complete(
        model="llama3.1-70b",
        prompt=[{"role": "user", "content": prompt}],
        session=session,
    )
    return _parse_judge_response(resp)


def _judge_plan(session, query: str, tool_used: str) -> tuple[float, str]:
    """Judge whether the agent selected the right tool."""
    from snowflake.cortex import complete

    prompt = (
        "You are an evaluation judge. Rate whether the SELECTED TOOL "
        "is the best choice for the QUERY on a scale from 0.0 to 1.0.\n\n"
        f"QUERY:\n{query}\n\n"
        f"SELECTED TOOL: {tool_used}\n\n"
        "Available tools:\n"
        "- documentation_search: Internal developer documentation "
        "including API rate limits, deployment processes, "
        "incident response, and onboarding\n"
        "- hr_policy_search: HR policies "
        "(PTO, benefits, remote work, expenses)\n"
        "- calculator: Mathematical calculations "
        "(arithmetic, percentages, time computations)\n\n"
        "Scoring:\n"
        "- 1.0: Clearly the best choice for this query.\n"
        "- 0.5-0.7: Reasonable but not optimal.\n"
        "- 0.0-0.3: Wrong tool - a different one is needed.\n\n"
        'Respond ONLY as JSON: {"score": <float>, "reason": "<brief>"}'
    )

    resp = complete(
        model="llama3.1-70b",
        prompt=[{"role": "user", "content": prompt}],
        session=session,
    )
    return _parse_judge_response(resp)


def _judge_act(session, answer: str, context: str) -> tuple[float, str]:
    """Judge whether the answer is faithful to source data."""
    from snowflake.cortex import complete

    if not context or context == "No relevant information found.":
        return 0.5, "No context retrieved (tool does not use retrieval)"

    prompt = (
        "You are an evaluation judge. Rate how faithfully the ANSWER "
        "reflects the SOURCE DATA on a scale from 0.0 to 1.0.\n\n"
        f"SOURCE DATA:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Scoring:\n"
        "- 1.0: Every claim is directly supported by the source.\n"
        "- 0.7-0.9: Most claims supported, minor extras.\n"
        "- 0.3-0.6: Some claims supported, significant fabrication.\n"
        "- 0.0-0.2: Mostly fabricated information.\n\n"
        'Respond ONLY as JSON: {"score": <float>, "reason": "<brief>"}'
    )

    resp = complete(
        model="llama3.1-70b",
        prompt=[{"role": "user", "content": prompt}],
        session=session,
    )
    return _parse_judge_response(resp)


def _parse_judge_response(resp: str) -> tuple[float, str]:
    """Parse the LLM judge JSON response."""
    try:
        text = resp.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            reason = data.get("reason", "No reason provided")
            return score, reason
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return 0.5, "Could not parse LLM judge response"


# --- Test Cases ---

TEST_CASES = [
    {
        "id": "case_1",
        "query": "What is 25% of 800?",
        "expected_tool": "calculator",
        "expected_answer": "200",
        "description": "Calculator: percentage (baseline)",
    },
    {
        "id": "case_2",
        "query": (
            "If our API handles 50 requests per second, "
            "how many requests can it handle in 30 minutes?"
        ),
        "expected_tool": "calculator",
        "expected_answer": "90,000 requests",
        "description": "Calculator: throughput computation",
    },
    {
        "id": "case_3",
        "query": "What is the Python indentation rule in our codebase?",
        "expected_tool": "documentation_search",
        "expected_answer": (
            "All Python code uses 2-space indentation. "
            "This is enforced by pre-commit hooks."
        ),
        "description": "Doc Search: coding standards (hallucination risk)",
    },
]


# --- Main ---


def _run_version(session, version: str) -> list[dict]:
    """Run all test cases for a given agent version.

    Args:
        session: Snowpark session.
        version: "v1" or "v2".

    Returns:
        List of result dicts.
    """
    print(f"\n  {'─' * 50}")
    print(f"  Agent {version.upper()}")
    print(f"  {'─' * 50}")

    results = []
    for case in TEST_CASES:
        print(f"\n  [{case['id']}] {case['description']}")

        agent = InternalDeveloperAssistant(version=version, snowpark_session=session)
        response = agent.run(case["query"])

        # Get context for Act evaluation
        tool_results = response.tool_output.get("results", [])
        context = (
            tool_results[0].get("content", "")
            if tool_results and "message" not in tool_results[0]
            else ""
        )
        # For calculator, use the result as context
        if response.tool_used == "calculator":
            result = response.tool_output.get("result")
            context = f"Calculation result: {result}" if result else ""

        # Evaluate with LLM judges
        goal_score, goal_reason = _judge_goal(
            session, case["query"], response.answer, case["expected_answer"]
        )
        plan_score, plan_reason = _judge_plan(
            session, case["query"], response.tool_used
        )
        act_score, act_reason = _judge_act(session, response.answer, context)

        print(f"    Tool: {response.tool_used}")
        print(f"    Answer: {response.answer[:60]}...")
        print(f"    Goal={goal_score:.2f} Plan={plan_score:.2f} Act={act_score:.2f}")

        results.append(
            {
                "case_id": case["id"],
                "query": case["query"],
                "description": case["description"],
                "version": version,
                "tool_used": response.tool_used,
                "expected_tool": case["expected_tool"],
                "answer": response.answer,
                "goal": goal_score,
                "goal_reason": goal_reason,
                "plan": plan_score,
                "plan_reason": plan_reason,
                "act": act_score,
                "act_reason": act_reason,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return results


def run_trulens_evaluation():
    """Run the AgentGPA evaluation for v1 and v2."""
    print("=" * 60)
    print("  AgentGPA Evaluation - Internal Developer Assistant")
    print("  Deboxx Poland Demo (v1 vs v2 Comparison)")
    print("=" * 60)

    # Connect
    print("\n[1/4] Connecting to Snowflake...")
    session = _create_snowpark_session()
    print(f"  Connected: {session.get_current_account()}")

    # Run v1
    print("\n[2/4] Evaluating v1 (weak prompts)...")
    v1_results = _run_version(session, "v1")

    # Run v2
    print("\n[3/4] Evaluating v2 (improved prompts)...")
    v2_results = _run_version(session, "v2")

    # Save combined results
    print("\n[4/4] Saving results...")
    all_results = {"v1": v1_results, "v2": v2_results}
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"  Saved to: {RESULTS_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("  COMPARISON: v1 vs v2")
    print("=" * 60)
    print()
    print(f"  {'Case':<10} {'':5} {'Goal':>6} {'Plan':>6} {'Act':>6} {'GPA':>6}")
    print(f"  {'─' * 45}")

    for v1, v2 in zip(v1_results, v2_results, strict=True):
        g1, p1, a1 = v1["goal"], v1["plan"], v1["act"]
        g2, p2, a2 = v2["goal"], v2["plan"], v2["act"]
        gpa1 = (g1 + p1 + a1) / 3
        gpa2 = (g2 + p2 + a2) / 3
        print(
            f"  {v1['case_id']:<10} v1:  {g1:>5.2f} {p1:>5.2f} {a1:>5.2f} {gpa1:>5.2f}"
        )
        print(f"  {'':10} v2:  {g2:>5.2f} {p2:>5.2f} {a2:>5.2f} {gpa2:>5.2f}")
        diff = gpa2 - gpa1
        arrow = "+" if diff > 0 else ""
        print(f"  {'':10} diff: {'':17} {arrow}{diff:.2f}")
        print()

    print("  View in Streamlit: uv run streamlit run app.py")
    print()

    session.close()


if __name__ == "__main__":
    run_trulens_evaluation()
