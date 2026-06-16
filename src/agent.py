"""Internal Developer Assistant - Agent.

Deboxx Poland Demo: AgentGPA Evaluation Demo.

Two modes:
    - "correct": Agent behaves properly (for Cases 1 & 3).
    - "broken":  Agent demonstrates failures (for Cases 2 & 4).
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
    """Internal Developer Assistant Agent for AgentGPA demo."""

    def __init__(self, mode: str = "correct"):
        """Initialize the agent.

        Args:
            mode: Either "correct" or "broken" to control behavior.
        """
        self.mode = mode
        self.tools = TOOLS
        self.last_response: AgentResponse | None = None

    def _select_tool(self, query: str) -> str:
        """Select the appropriate tool for the query."""
        query_lower = query.lower()

        if self.mode == "broken":
            return self._select_tool_broken(query_lower)

        math_indicators = [
            "calculate",
            "how much",
            "what is",
            "%",
            "percent",
            "sum",
            "multiply",
            "divide",
        ]
        hr_indicators = [
            "pto",
            "vacation",
            "benefits",
            "remote",
            "expense",
            "policy",
            "employee",
            "time off",
            "sick",
            "leave",
        ]

        has_numbers = any(c.isdigit() for c in query)
        has_math_keyword = any(ind in query_lower for ind in math_indicators)

        if has_numbers and has_math_keyword:
            return "calculator"

        if any(ind in query_lower for ind in hr_indicators):
            return "hr_policy_search"

        return "documentation_search"

    def _select_tool_broken(self, query_lower: str) -> str:
        """Intentionally broken tool selection for demo."""
        # Case 4: Math question routed to HR search instead of calculator
        if any(c.isdigit() for c in query_lower) and "%" in query_lower:
            return "hr_policy_search"

        # Case 2: still uses correct tool; hallucination is in generation
        return "hr_policy_search"

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
            if "%" in query and len(numbers) >= 2:
                expression = f"{numbers[0]}/100*{numbers[1]}"
            else:
                expression = query
            return tool_fn(expression)
        return tool_fn(query)

    def _generate_answer(self, query: str, tool_name: str, tool_output: dict) -> str:
        """Generate answer from tool output."""
        if self.mode == "broken":
            return self._generate_answer_broken(query, tool_name, tool_output)

        if tool_name == "calculator":
            result = tool_output.get("result")
            if result is not None:
                if "$" in query:
                    return f"The answer is ${result:.2f}."
                return f"The answer is {result}."
            error = tool_output.get("error", "unknown error")
            return f"I couldn't calculate that: {error}"

        if tool_name == "hr_policy_search":
            results = tool_output.get("results", [])
            if results and "message" not in results[0]:
                return f"According to our HR policy: {results[0]['content']}"
            return "I couldn't find a relevant HR policy for your question."

        if tool_name == "documentation_search":
            results = tool_output.get("results", [])
            if results and "message" not in results[0]:
                return f"From our documentation: {results[0]['content']}"
            return "I couldn't find relevant documentation for your question."

        return "I'm sorry, I don't have enough information to answer that."

    def _generate_answer_broken(
        self, query: str, tool_name: str, tool_output: dict
    ) -> str:
        """Generate intentionally broken answers for demo."""
        query_lower = query.lower()

        # Case 2: "vacation days" - hallucinate information not in source
        if "vacation" in query_lower:
            return (
                "According to our policy, employees receive "
                "14 vacation days per year, plus 7 additional "
                "personal days. Vacation days increase to 21 "
                "after 4 years of service and 28 after 6 years."
            )

        # Case 4: Math question routed to HR (wrong tool)
        if "%" in query and any(c.isdigit() for c in query):
            results = tool_output.get("results", [])
            if results and "message" not in results[0]:
                content = results[0]["content"]
                return f"Based on our HR policies: {content}"
            return (
                "I couldn't find relevant information "
                "in our HR policies for this calculation."
            )

        return "I don't have enough information to answer that."

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
        tool_name = self._select_tool(query)
        tool_output = self._execute_tool(query, tool_name)
        answer = self._generate_answer(query, tool_name, tool_output)

        # Store full response for inspection
        tool_input = query
        if tool_name == "calculator":
            numbers = re.findall(r"[\d.]+", query)
            if "%" in query and len(numbers) >= 2:
                tool_input = f"{numbers[0]}/100*{numbers[1]}"

        self.last_response = AgentResponse(
            query=query,
            tool_used=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            answer=answer,
            reasoning=(
                f"Selected tool: {tool_name} | "
                f"Reason: {self.tools[tool_name]['description']} | "
                f"Mode: {self.mode}"
            ),
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
