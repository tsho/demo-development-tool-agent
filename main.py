"""Quick demo runner - shows all 3 cases interactively."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from src.agent import InternalDeveloperAssistant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def demo():
    """Run all 3 demo cases and log results."""
    logger.info("\n" + "=" * 60)
    logger.info("  Internal Developer Assistant")
    logger.info("=" * 60)

    cases = [
        "What is the API rate limit?",
        "What is the Python indentation rule in our codebase?",
        (
            "If our API handles 50 requests per second, "
            "how many requests can it handle in 30 minutes?"
        ),
    ]

    for i, query in enumerate(cases, 1):
        agent = InternalDeveloperAssistant(version="v2")
        response = agent.run(query)

        logger.info("\n%s", "─" * 60)
        logger.info("  Case %d", i)
        logger.info("  User: %s", query)
        logger.info("  Tool: %s", response.tool_used)
        logger.info("  Agent: %s", response.answer)
        logger.info("─" * 60)


if __name__ == "__main__":
    demo()
