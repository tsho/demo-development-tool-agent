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


def _create_session():
    """Create Snowpark session for Cortex."""
    import os

    from snowflake.snowpark import Session

    return Session.builder.configs(
        {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "role": os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        }
    ).create()


def demo():
    """Run all 3 demo cases and log results."""
    logger.info("\n" + "=" * 60)
    logger.info("  Internal Developer Assistant - Deboxx Poland Demo")
    logger.info("=" * 60)

    session = _create_session()

    cases = [
        "What is the API rate limit?",
        "What is the deployment process?",
        (
            "If our API handles 50 requests per second, "
            "how many requests can it handle in 30 minutes?"
        ),
    ]

    for i, query in enumerate(cases, 1):
        agent = InternalDeveloperAssistant(version="v2", snowpark_session=session)
        response = agent.run(query)

        logger.info("\n%s", "─" * 60)
        logger.info("  Case %d", i)
        logger.info("  User: %s", query)
        logger.info("  Tool: %s", response.tool_used)
        logger.info("  Agent: %s", response.answer)
        logger.info("─" * 60)

    session.close()


if __name__ == "__main__":
    demo()
