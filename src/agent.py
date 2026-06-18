"""Internal Developer Assistant - LLM-based Agent.

Deboxx Poland Demo: AgentGPA Evaluation Demo.

Uses Snowflake Cortex LLM for tool selection and answer generation.
v1/v2 behavior is controlled by different system prompts.
"""

import json
import re
from dataclasses import dataclass

from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

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


class InternalDeveloperAssistant:
    """LLM-based Internal Developer Assistant for AgentGPA demo.

    Uses Cortex LLM for both tool selection and answer generation.
    The version (v1/v2) determines which prompts are used.
    """

    def __init__(self, version: str = "v2", snowpark_session=None):
        """Initialize the agent.

        Args:
            version: "v1" (weak prompts) or "v2" (improved prompts).
            snowpark_session: Snowpark session for Cortex calls.
        """
        self.version = version
        self.tools = TOOLS
        self.session = snowpark_session
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

    def _select_tool_llm(self, query: str) -> str:
        """Use LLM to select the appropriate tool.

        Args:
            query: The user's question.

        Returns:
            Tool name string.
        """
        from snowflake.cortex import complete

        tool_names = ", ".join(self.tools.keys())
        prompt = self._tool_selection_prompt.format(tool_names=tool_names)

        resp = complete(
            model="llama3.1-70b",
            prompt=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            session=self.session,
        )

        # Parse tool name from response
        selected = resp.strip().lower().replace("'", "").replace('"', "")
        for tool_name in self.tools:
            if tool_name in selected:
                return tool_name

        # Fallback to documentation_search
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
        },
    )
    def _execute_tool(self, query: str, tool_name: str) -> dict:
        """Execute the selected tool and return raw output.

        Args:
            query: The user's query.
            tool_name: Name of the tool to execute.

        Returns:
            Tool output dictionary.
        """
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
    ) -> str:
        """Use LLM to generate an answer from tool output.

        Args:
            query: The user's question.
            tool_name: The tool that was used.
            tool_output: Raw output from the tool.

        Returns:
            Generated answer string.
        """
        from snowflake.cortex import complete

        # Build context from tool output
        if tool_name == "calculator":
            # Calculator results are deterministic, no LLM needed
            result = tool_output.get("result")
            if result is not None:
                return f"The answer is {result:,.2f}."
            return f"Calculator error: {tool_output.get('error')}"

        results = tool_output.get("results", [])
        if results and "message" not in results[0]:
            context = results[0]["content"]
        else:
            context = "No relevant information found."

        prompt = self._answer_generation_prompt.format(context=context, query=query)

        resp = complete(
            model="llama3.1-70b",
            prompt=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            session=self.session,
        )

        return resp.strip()

    @instrument(
        span_type=SpanAttributes.SpanType.AGENT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        """Process a user query (TruLens-compatible entry point).

        Args:
            query: The user's question.

        Returns:
            The agent's answer as a string.
        """
        tool_name = self._select_tool_llm(query)
        tool_output = self._execute_tool(query, tool_name)
        answer = self._generate_answer_llm(query, tool_name, tool_output)

        self.last_response = AgentResponse(
            query=query,
            tool_used=tool_name,
            tool_input=query,
            tool_output=tool_output,
            answer=answer,
            reasoning=(f"Selected tool: {tool_name} | Version: {self.version}"),
        )
        return answer

    def run(self, query: str) -> AgentResponse:
        """Process a user query and return full response.

        Args:
            query: The user's question.

        Returns:
            AgentResponse with tool usage details and answer.
        """
        self.query(query)
        return self.last_response  # type: ignore[return-value]
