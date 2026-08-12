"""Internal Developer Assistant - LLM-based Agent.

Uses an OpenAI-compatible LLM API for tool selection and answer generation.
v1/v2 behavior is controlled by different system prompts.
Provider is configured via LLM_PROVIDER env var (cortex/openai/anthropic).
"""

import json
import re
import time
from dataclasses import dataclass, field

from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from src.llm_client import chat_complete_with_usage
from src.tools import TOOLS


@dataclass
class AgentResponse:
    """Response object returned by the agent."""

    query: str
    tool_used: str
    tool_input: str
    tool_output: dict
    answer: str
    reasoning: str
    # Observability fields
    step_timings: dict = field(default_factory=dict)
    selection_raw_response: str = ""
    prompts: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=dict)


class InternalDeveloperAssistant:
    """LLM-based Internal Developer Assistant for AgentGPA demo.

    Uses an OpenAI-compatible API for both tool selection and answer generation.
    The version (v1/v2) determines which prompts and data are used.
    """

    def __init__(self, version: str = "v2"):
        """Initialize the agent.

        Args:
            version: "v1" (weak prompts) or "v2" (improved prompts).
        """
        self.version = version
        self.tools = TOOLS
        self.last_response: AgentResponse | None = None

        if version == "v1":
            from src.prompts_v1 import (
                ANSWER_GENERATION_PROMPT,
                TOOL_SELECTION_PROMPT,
            )
        else:
            from src.prompts_v2 import (
                ANSWER_GENERATION_PROMPT,
                TOOL_SELECTION_PROMPT,
            )

        self._tool_selection_prompt = TOOL_SELECTION_PROMPT
        self._answer_generation_prompt = ANSWER_GENERATION_PROMPT
        self._doc_file = (
            "documentation.json" if version == "v1" else "documentation_v2.json"
        )

    @instrument(
        span_type=SpanAttributes.SpanType.TOOL,
        attributes=lambda ret, exception, *args, **kwargs: {
            "tool_selection.selected_tool": ret,
            "tool_selection.query": (
                kwargs.get("query") or (args[1] if len(args) > 1 else "")
            ),
        },
    )
    def _select_tool_llm(self, query: str) -> str:
        """Use LLM to select the appropriate tool."""
        tool_names = ", ".join(self.tools.keys())
        prompt = self._tool_selection_prompt.format(tool_names=tool_names)

        resp, usage = chat_complete_with_usage(system_prompt=prompt, user_message=query)

        # Store raw response and usage for observability
        self._last_selection_raw = resp
        self._last_selection_prompt = prompt
        self._last_selection_usage = usage

        selected = resp.strip().lower().replace("'", "").replace('"', "")
        for tool_name in self.tools:
            if tool_name in selected:
                return tool_name
        return "documentation_search"

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes=lambda ret, exception, *args, **kwargs: {
            SpanAttributes.RETRIEVAL.QUERY_TEXT: (
                kwargs.get("query") or (args[1] if len(args) > 1 else None)
            ),
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: (
                [json.dumps(ret.get("results", []))] if ret else []
            ),
            "tool.name": (
                kwargs.get("tool_name") or (args[2] if len(args) > 2 else "unknown")
            ),
        },
    )
    def _execute_tool(self, query: str, tool_name: str) -> dict:
        """Execute the selected tool and return raw output."""
        tool_fn = self.tools[tool_name]["function"]
        if tool_name == "calculator":
            numbers = re.findall(r"[\d.]+", query)
            query_lower = query.lower()
            if "%" in query and len(numbers) >= 2:
                expression = f"{numbers[0]}/100*{numbers[1]}"
            elif "per second" in query_lower and "minute" in query_lower:
                expression = f"{numbers[0]}*{numbers[1]}*60"
            elif len(numbers) >= 2:
                expression = "*".join(numbers)
            else:
                expression = query
            return tool_fn(expression)
        if tool_name == "documentation_search":
            return tool_fn(query, data_file=self._doc_file)
        return tool_fn(query)

    def _generate_answer_llm(
        self, query: str, tool_name: str, tool_output: dict
    ) -> tuple[str, str, dict]:
        """Use LLM to generate an answer. Returns (answer, prompt_used, usage)."""
        if tool_name == "calculator":
            result = tool_output.get("result")
            if result is not None:
                return f"The answer is {result:,.2f}.", "", {}
            return f"Calculator error: {tool_output.get('error')}", "", {}

        results = tool_output.get("results", [])
        if results and "message" not in results[0]:
            context = results[0]["content"]
        else:
            context = "No relevant information found."

        prompt = self._answer_generation_prompt.format(
            context=context, query=query
        )

        resp, usage = chat_complete_with_usage(system_prompt=prompt, user_message=query)
        return resp.strip(), prompt, usage

    @instrument(
        span_type=SpanAttributes.SpanType.AGENT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        """Process a user query (TruLens-compatible entry point)."""
        t0 = time.perf_counter()
        tool_name = self._select_tool_llm(query)
        t1 = time.perf_counter()

        tool_output = self._execute_tool(query, tool_name)
        t2 = time.perf_counter()

        answer, answer_prompt, answer_usage = self._generate_answer_llm(
            query, tool_name, tool_output
        )
        t3 = time.perf_counter()

        self.last_response = AgentResponse(
            query=query,
            tool_used=tool_name,
            tool_input=query,
            tool_output=tool_output,
            answer=answer,
            reasoning=f"Selected tool: {tool_name} | Version: {self.version}",
            step_timings={
                "tool_selection_s": round(t1 - t0, 3),
                "tool_execution_s": round(t2 - t1, 3),
                "answer_generation_s": round(t3 - t2, 3),
                "total_s": round(t3 - t0, 3),
            },
            selection_raw_response=getattr(self, "_last_selection_raw", ""),
            prompts={
                "tool_selection": getattr(self, "_last_selection_prompt", ""),
                "answer_generation": answer_prompt,
            },
            token_usage={
                "tool_selection": getattr(self, "_last_selection_usage", {}),
                "answer_generation": answer_usage,
            },
        )
        return answer

    def run(self, query: str) -> AgentResponse:
        """Process a user query and return full response."""
        self.query(query)
        return self.last_response  # type: ignore[return-value]
