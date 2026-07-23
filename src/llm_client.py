"""Unified LLM client - OpenAI-compatible REST API.

Supports Snowflake Cortex REST API (default), OpenAI, and Anthropic.
Provider is selected via LLM_PROVIDER environment variable.
"""

import logging
import os
import tomllib
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def _load_snowflake_connection() -> dict:
    """Load Snowflake connection from ~/.snowflake/connections.toml."""
    toml_path = Path.home() / ".snowflake" / "connections.toml"
    if not toml_path.exists():
        return {}

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    conn_name = os.getenv(
        "SNOWFLAKE_CONNECTION_NAME",
        config.get("default_connection_name", ""),
    )
    if not conn_name:
        return {}

    return config.get(conn_name, {})


def _complete_cortex(system_prompt: str, user_message: str) -> str:
    """Call Snowflake Cortex REST API (OpenAI-compatible)."""
    conn = _load_snowflake_connection()
    account = os.getenv("SNOWFLAKE_ACCOUNT", conn.get("account", ""))
    token = os.getenv("SNOWFLAKE_TOKEN", conn.get("token", ""))

    if not account or not token:
        raise ValueError(
            "Cortex requires SNOWFLAKE_ACCOUNT and SNOWFLAKE_TOKEN "
            "(or ~/.snowflake/connections.toml with token)."
        )

    model = os.getenv("LLM_MODEL", "llama3.1-70b")
    url = (
        f"https://{account}.snowflakecomputing.com"
        "/api/v2/cortex/inference:complete"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": 2048,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def _complete_openai(system_prompt: str, user_message: str) -> str:
    """Call OpenAI-compatible API."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for openai provider.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o")
    url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": 2048,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def _complete_anthropic(system_prompt: str, user_message: str) -> str:
    """Call Anthropic Messages API."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for anthropic provider.")

    model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": 0,
        "max_tokens": 2048,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    return result["content"][0]["text"].strip()


def chat_complete(system_prompt: str, user_message: str) -> str:
    """Call LLM with system + user messages, return content string.

    Provider is selected via LLM_PROVIDER env var:
        - "cortex" (default): Snowflake Cortex REST API
        - "openai": OpenAI or any OpenAI-compatible endpoint
        - "anthropic": Anthropic Messages API

    Args:
        system_prompt: System/instruction prompt.
        user_message: User message content.

    Returns:
        LLM response content as a string.
    """
    provider = os.getenv("LLM_PROVIDER", "cortex")

    if provider == "cortex":
        return _complete_cortex(system_prompt, user_message)
    if provider == "openai":
        return _complete_openai(system_prompt, user_message)
    if provider == "anthropic":
        return _complete_anthropic(system_prompt, user_message)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Use 'cortex', 'openai', or 'anthropic'."
    )
