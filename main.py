"""Quick demo runner - shows all 4 cases interactively."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent import InternalDeveloperAssistant  # noqa: E402


def demo():
    """Run all 4 demo cases and print results."""
    print("\n" + "=" * 60)
    print("  Internal Developer Assistant - Deboxx Poland Demo")
    print("=" * 60)

    cases = [
        ("correct", "How many PTO days do employees receive?"),
        ("broken", "How many vacation days do employees receive?"),
        ("correct", "What is 15% of $1200?"),
        ("broken", "What is 15% of $1200?"),
    ]

    for i, (mode, query) in enumerate(cases, 1):
        agent = InternalDeveloperAssistant(mode=mode)
        response = agent.run(query)

        print(f"\n{'─' * 60}")
        print(f"  Case {i} (mode={mode})")
        print(f"  User: {query}")
        print(f"  Tool: {response.tool_used}")
        print(f"  Agent: {response.answer}")
        print(f"{'─' * 60}")

    print()


if __name__ == "__main__":
    demo()
