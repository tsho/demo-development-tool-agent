"""TruLens Evaluation - Deboxx Poland Demo.

Evaluates the Internal Developer Assistant using TruLens with Snowflake Cortex.
Uses Cortex LLM functions as the judge for Groundedness and Answer Relevance.

Reference: https://www.trulens.org/cookbook/models/snowflake_cortex/cortex_llm_quickstart/

Metrics:
    - Groundedness: Is the answer grounded in retrieved context?
    - Answer Relevance: Does the answer address the user's question?
    - Tool Selection Accuracy: Did the agent use the correct tool?
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from snowflake.snowpark import Session  # noqa: E402
from trulens.apps.app import TruApp  # noqa: E402
from trulens.core import Metric, Selector  # noqa: E402
from trulens.core.session import TruSession  # noqa: E402
from trulens.providers.cortex import Cortex  # noqa: E402

from src.agent import InternalDeveloperAssistant  # noqa: E402

# --- Snowflake Connection ---


def _create_snowpark_session() -> Session:
    """Create a Snowflake Snowpark session from environment variables.

    Required env vars:
        SNOWFLAKE_ACCOUNT
        SNOWFLAKE_USER
        SNOWFLAKE_USER_PASSWORD

    Optional env vars:
        SNOWFLAKE_ROLE (default: ACCOUNTADMIN)
        SNOWFLAKE_DATABASE
        SNOWFLAKE_SCHEMA
        SNOWFLAKE_WAREHOUSE
    """
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_USER_PASSWORD"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print(
            "  Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, "
            "SNOWFLAKE_USER_PASSWORD to use Cortex."
        )
        sys.exit(1)

    connection_params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
        "role": os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
    }

    return Session.builder.configs(connection_params).create()


# --- Test Cases ---

TEST_CASES = [
    {
        "id": "case_1",
        "query": "How many PTO days do employees receive?",
        "mode": "correct",
        "expected_tool": "hr_policy_search",
        "description": "SUCCESS - Correct tool + faithful answer",
    },
    {
        "id": "case_2",
        "query": "How many vacation days do employees receive?",
        "mode": "broken",
        "expected_tool": "hr_policy_search",
        "description": "FAILURE - Hallucination (Faithfulness drops)",
    },
    {
        "id": "case_3",
        "query": "What is 15% of $1200?",
        "mode": "correct",
        "expected_tool": "calculator",
        "description": "SUCCESS - Correct tool (Calculator)",
    },
    {
        "id": "case_4",
        "query": "What is 15% of $1200?",
        "mode": "broken",
        "expected_tool": "calculator",
        "description": "FAILURE - Wrong tool (Tool Selection drops)",
    },
]


# --- Tool Selection (deterministic) ---


def _evaluate_tool_selection(test_cases_results: list[dict]) -> None:
    """Print deterministic tool selection accuracy.

    Args:
        test_cases_results: List of {case_id, tool_used, expected_tool}.
    """
    print("\n" + "=" * 60)
    print("  Tool Selection Accuracy (Deterministic)")
    print("=" * 60)
    print()

    correct = 0
    for r in test_cases_results:
        match = r["tool_used"] == r["expected_tool"]
        correct += int(match)
        status = "PASS" if match else "FAIL"
        print(
            f"  {r['case_id']}: {status} "
            f"(used={r['tool_used']}, expected={r['expected_tool']})"
        )

    accuracy = correct / len(test_cases_results)
    print(f"\n  Accuracy: {accuracy:.0%} ({correct}/{len(test_cases_results)})")
    print()


# --- Main ---


def run_trulens_evaluation():
    """Run the TruLens evaluation pipeline with Cortex as judge."""
    print("=" * 60)
    print("  TruLens Evaluation - Internal Developer Assistant")
    print("  Deboxx Poland Demo (Cortex LLM Judge)")
    print("=" * 60)
    print()

    # Create Snowpark session
    print("[1/4] Connecting to Snowflake...")
    snowpark_session = _create_snowpark_session()
    print(f"  Connected: {snowpark_session.get_current_account()}")

    # Initialize Cortex provider
    print("[2/4] Initializing Cortex provider...")
    provider = Cortex(
        snowpark_session=snowpark_session,
        model_engine="llama3.1-8b",
    )
    print("  Provider: Cortex (llama3.1-8b)")

    # Initialize TruLens session
    print("[3/4] Setting up TruLens session...")
    session = TruSession()
    session.reset_database()

    # Define metrics
    # groundedness: source (retrieved context) + statement (agent output)
    # relevance: prompt (user input) + response (agent output)
    f_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="Groundedness",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        },
    )

    f_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="Answer Relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    # Run test cases
    print("[4/4] Running test cases...")
    tool_selection_results = []

    for case in TEST_CASES:
        print(f"\n  Running: {case['id']} - {case['description']}")

        agent = InternalDeveloperAssistant(mode=case["mode"])

        tru_app = TruApp(
            agent,
            app_name="Internal Developer Assistant",
            app_version=case["mode"],
            feedbacks=[f_groundedness, f_answer_relevance],
            main_method=agent.query,
        )

        with tru_app:
            answer = agent.query(case["query"])

        print(f"    Answer: {answer[:70]}...")

        # Track tool selection for deterministic eval
        tool_selection_results.append(
            {
                "case_id": case["id"],
                "tool_used": agent.last_response.tool_used,
                "expected_tool": case["expected_tool"],
            }
        )

    # Print tool selection results
    _evaluate_tool_selection(tool_selection_results)

    # Wait for async feedback evaluation to complete
    print("=" * 60)
    print("  Waiting for LLM-as-Judge evaluations...")
    print("=" * 60)
    print()

    import time

    for i in range(30):
        time.sleep(2)
        leaderboard = session.get_leaderboard()
        if not leaderboard.empty and leaderboard.shape[1] > 2:
            break
        print(f"  ... waiting ({(i + 1) * 2}s)")

    # Print TruLens leaderboard
    print("=" * 60)
    print("  TruLens Leaderboard (Cortex LLM-as-Judge)")
    print("=" * 60)
    print()

    leaderboard = session.get_leaderboard()
    if leaderboard.empty:
        print("  (Results still processing - launch dashboard to view)")
    else:
        print(leaderboard.to_string())
    print()

    print("  Launch dashboard: uv run trulens-dashboard")
    print()

    # Cleanup
    snowpark_session.close()


if __name__ == "__main__":
    run_trulens_evaluation()
