"""AgentGPA Evaluation - Deboxx Poland Demo.

Evaluates the Internal Developer Assistant on three metrics:
    - Faithfulness: Is the answer grounded in the retrieved data?
    - Tool Selection Accuracy: Did the agent use the correct tool?
    - Relevancy: Is the answer relevant to the question?
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import AgentResponse, InternalDeveloperAssistant  # noqa: E402

# --- Test Cases ---

TEST_CASES = [
    {
        "id": "case_1",
        "query": "How many PTO days do employees receive?",
        "expected_tool": "hr_policy_search",
        "expected_answer_contains": "20",
        "ground_truth_source": "pto-001",
        "mode": "correct",
        "description": "SUCCESS - Correct tool + faithful answer",
    },
    {
        "id": "case_2",
        "query": "How many vacation days do employees receive?",
        "expected_tool": "hr_policy_search",
        "expected_answer_contains": "20",
        "ground_truth_source": "pto-001",
        "mode": "broken",
        "description": "FAILURE - Hallucination (Faithfulness drops)",
    },
    {
        "id": "case_3",
        "query": "What is 15% of $1200?",
        "expected_tool": "calculator",
        "expected_answer_contains": "180",
        "ground_truth_source": None,
        "mode": "correct",
        "description": "SUCCESS - Correct tool (Calculator) + correct result",
    },
    {
        "id": "case_4",
        "query": "What is 15% of $1200?",
        "expected_tool": "calculator",
        "expected_answer_contains": "180",
        "ground_truth_source": None,
        "mode": "broken",
        "description": "FAILURE - Wrong tool (HR Search instead of Calculator)",
    },
]


# --- Evaluation Metrics ---


def evaluate_faithfulness(response: AgentResponse, test_case: dict) -> dict:
    """Evaluate if the answer is grounded in the tool's output.

    Args:
        response: The agent's response.
        test_case: The test case definition.

    Returns:
        Dict with score (0.0-1.0) and reason.
    """
    tool_output = response.tool_output
    answer = response.answer.lower()

    if response.tool_used == "calculator":
        result = tool_output.get("result")
        if result is not None and str(int(result)) in answer:
            return {"score": 1.0, "reason": "Answer matches calculator output"}
        if result is not None:
            return {
                "score": 0.5,
                "reason": "Calculator produced result but answer doesn't reflect it",
            }
        return {
            "score": 0.0,
            "reason": "Calculator failed and answer is not grounded",
        }

    results = tool_output.get("results", [])
    if not results or "message" in results[0]:
        if "couldn't find" in answer:
            return {
                "score": 1.0,
                "reason": "Correctly reported no results found",
            }
        return {
            "score": 0.0,
            "reason": ("No source data but agent generated an answer (hallucination)"),
        }

    source_content = results[0].get("content", "").lower()

    numbers_in_answer = re.findall(r"\d+", answer)
    numbers_in_source = re.findall(r"\d+", source_content)

    if numbers_in_answer:
        grounded_numbers = [n for n in numbers_in_answer if n in numbers_in_source]
        if len(grounded_numbers) == len(numbers_in_answer):
            return {
                "score": 1.0,
                "reason": "All factual claims are grounded in source data",
            }
        if len(grounded_numbers) > 0:
            ratio = len(grounded_numbers) / len(numbers_in_answer)
            return {
                "score": ratio,
                "reason": (
                    f"Only {len(grounded_numbers)}/"
                    f"{len(numbers_in_answer)} claims are grounded"
                ),
            }
        return {
            "score": 0.0,
            "reason": (
                "Answer contains facts not present in source data (HALLUCINATION)"
            ),
        }

    return {
        "score": 0.8,
        "reason": "Answer appears generally grounded (no numeric claims to verify)",
    }


def evaluate_tool_selection(response: AgentResponse, test_case: dict) -> dict:
    """Evaluate if the agent picked the correct tool.

    Args:
        response: The agent's response.
        test_case: The test case definition.

    Returns:
        Dict with score (0.0 or 1.0) and reason.
    """
    expected = test_case["expected_tool"]
    actual = response.tool_used

    if actual == expected:
        return {
            "score": 1.0,
            "reason": f"Correctly selected '{actual}'",
        }
    return {
        "score": 0.0,
        "reason": (f"Selected '{actual}' but expected '{expected}' (WRONG TOOL)"),
    }


def evaluate_relevancy(response: AgentResponse, test_case: dict) -> dict:
    """Evaluate if the answer addresses the user's question.

    Args:
        response: The agent's response.
        test_case: The test case definition.

    Returns:
        Dict with score (0.0-1.0) and reason.
    """
    expected_content = test_case["expected_answer_contains"].lower()
    answer = response.answer.lower()
    query_lower = test_case["query"].lower()

    has_expected = expected_content in answer

    if "%" in query_lower or "calculate" in query_lower:
        is_topical = any(w in answer for w in ["answer", "result", "equal", "$", "is"])
    elif "pto" in query_lower or "vacation" in query_lower or "days" in query_lower:
        is_topical = any(
            w in answer for w in ["days", "pto", "policy", "vacation", "time off"]
        )
    else:
        is_topical = True

    if has_expected and is_topical:
        return {
            "score": 1.0,
            "reason": "Answer directly addresses the question with correct info",
        }
    if is_topical and not has_expected:
        return {
            "score": 0.5,
            "reason": "Answer is topically relevant but provides incorrect info",
        }
    if has_expected and not is_topical:
        return {
            "score": 0.5,
            "reason": "Contains expected info but framing is off-topic",
        }
    return {
        "score": 0.0,
        "reason": "Answer is not relevant to the question (IRRELEVANT)",
    }


# --- Main Evaluation Runner ---


def run_evaluation():
    """Run all test cases and produce AgentGPA-compatible evaluation results."""
    print("=" * 70)
    print("  AgentGPA Evaluation - Internal Developer Assistant")
    print("  Deboxx Poland Demo")
    print("=" * 70)
    print()

    results = []

    for test_case in TEST_CASES:
        agent = InternalDeveloperAssistant(mode=test_case["mode"])
        response = agent.run(test_case["query"])

        faithfulness = evaluate_faithfulness(response, test_case)
        tool_selection = evaluate_tool_selection(response, test_case)
        relevancy = evaluate_relevancy(response, test_case)

        result = {
            "case_id": test_case["id"],
            "description": test_case["description"],
            "query": test_case["query"],
            "mode": test_case["mode"],
            "tool_used": response.tool_used,
            "answer": response.answer,
            "metrics": {
                "faithfulness": faithfulness,
                "tool_selection": tool_selection,
                "relevancy": relevancy,
            },
        }
        results.append(result)

        _print_case_result(test_case, response, faithfulness, tool_selection, relevancy)

    _print_summary(results)

    output_path = Path(__file__).parent / "results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  Results exported to: {output_path}")
    print()

    return results


def _print_case_result(test_case, response, faithfulness, tool_selection, relevancy):
    """Print results for a single test case."""
    print(f"{'─' * 70}")
    print(f"  {test_case['id'].upper()}: {test_case['description']}")
    print(f"{'─' * 70}")
    print(f"  Query:  {test_case['query']}")
    expected = test_case["expected_tool"]
    print(f"  Tool:   {response.tool_used} (expected: {expected})")
    print(f"  Answer: {response.answer[:80]}...")
    print()
    print("  Metrics:")
    f_score = faithfulness["score"]
    f_reason = faithfulness["reason"]
    print(f"    Faithfulness:     {f_score:.1f}  - {f_reason}")
    t_score = tool_selection["score"]
    t_reason = tool_selection["reason"]
    print(f"    Tool Selection:   {t_score:.1f}  - {t_reason}")
    r_score = relevancy["score"]
    r_reason = relevancy["reason"]
    print(f"    Relevancy:        {r_score:.1f}  - {r_reason}")
    print()


def _print_summary(results):
    """Print summary table and key takeaways."""
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    header = (
        f"  {'Case':<10} {'Faithful':>10} "
        f"{'Tool Sel.':>10} {'Relevancy':>10} {'Avg':>8}"
    )
    print(header)
    print(f"  {'─' * 48}")

    total_f, total_t, total_r = 0.0, 0.0, 0.0
    for r in results:
        f = r["metrics"]["faithfulness"]["score"]
        t = r["metrics"]["tool_selection"]["score"]
        v = r["metrics"]["relevancy"]["score"]
        avg = (f + t + v) / 3
        total_f += f
        total_t += t
        total_r += v
        status = "pass" if avg >= 0.8 else "FAIL"
        print(
            f"  {r['case_id']:<10} {f:>10.1f} {t:>10.1f} "
            f"{v:>10.1f} {avg:>7.2f}  {status}"
        )

    n = len(results)
    print(f"  {'─' * 48}")
    avg_all = (total_f + total_t + total_r) / (n * 3)
    print(
        f"  {'AVERAGE':<10} {total_f / n:>10.2f} "
        f"{total_t / n:>10.2f} {total_r / n:>10.2f} {avg_all:>7.2f}"
    )
    print()

    print("  KEY DEMO POINTS:")
    print("  -----------------")
    print(
        "  - Case 1 vs 2: Same tool, different output "
        "-> Faithfulness catches hallucination"
    )
    print(
        "  - Case 3 vs 4: Same query, different tool  "
        "-> Tool Selection catches wrong routing"
    )
    print(
        "  - All cases:   Relevancy evaluates answer-question alignment independently"
    )
    print()


if __name__ == "__main__":
    run_evaluation()
