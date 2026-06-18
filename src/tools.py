"""Internal Developer Assistant - Tools.

Deboxx Poland Demo: AgentGPA Evaluation Demo.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def documentation_search(query: str) -> dict:
    """Search internal developer documentation."""
    docs = json.loads((DATA_DIR / "documentation.json").read_text())
    query_lower = query.lower()
    stop_words = {"what", "is", "the", "a", "an", "how", "can", "it", "in", "our", "if"}
    keywords = [
        kw for kw in query_lower.split() if kw not in stop_words and len(kw) > 2
    ]
    scored = []
    for doc in docs:
        text = (doc["title"] + " " + doc["content"]).lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, doc))
    # Sort by relevance (most keyword matches first)
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [doc for _, doc in scored]
    no_results = [{"message": "No relevant documentation found."}]
    return {
        "tool": "documentation_search",
        "query": query,
        "results": results if results else no_results,
    }


def hr_policy_search(query: str) -> dict:
    """Search HR policies and employee handbook."""
    policies = json.loads((DATA_DIR / "hr_policies.json").read_text())
    query_lower = query.lower()
    keywords = query_lower.split()
    results = []
    for policy in policies:
        text = (policy["title"] + " " + policy["content"]).lower()
        if any(kw in text for kw in keywords):
            results.append(policy)
    no_results = [{"message": "No relevant HR policy found."}]
    return {
        "tool": "hr_policy_search",
        "query": query,
        "results": results if results else no_results,
    }


def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression safely."""
    allowed_chars = set("0123456789+-*/.() %")
    if not all(c in allowed_chars for c in expression.replace(" ", "")):
        return {
            "tool": "calculator",
            "expression": expression,
            "error": "Invalid characters in expression",
        }

    clean_expr = expression.replace("%", "/100*")
    try:
        result = eval(clean_expr)  # noqa: S307
        return {
            "tool": "calculator",
            "expression": expression,
            "result": result,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "tool": "calculator",
            "expression": expression,
            "error": str(e),
        }


TOOLS = {
    "documentation_search": {
        "function": documentation_search,
        "description": (
            "Search internal developer documentation for "
            "technical guides, API references, and processes."
        ),
    },
    "hr_policy_search": {
        "function": hr_policy_search,
        "description": (
            "Search HR policies including PTO, benefits, "
            "remote work, and expense policies."
        ),
    },
    "calculator": {
        "function": calculator,
        "description": (
            "Perform mathematical calculations. "
            "Supports basic arithmetic and percentages."
        ),
    },
}
