"""Extract tool_use blocks from LLM response content.

Handles both Anthropic-style content block lists and OpenAI-style
tool_calls arrays. Works with raw dicts — no SDK dependency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


class ToolUseExtractorError(Exception):
    """Raised on invalid input that cannot be parsed."""


@dataclass
class ToolUse:
    """A single tool_use block extracted from a response.

    Attributes:
        id: unique tool call ID (e.g. "call_abc123" or "toolu_xyz").
        name: tool name (e.g. "web_search").
        input: dict of arguments passed to the tool.
        raw: the original block dict (deep copy).
    """

    id: str
    name: str
    input: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Shorthand to get an argument from input."""
        return self.input.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.input


def extract_tool_uses(content: Any) -> list[ToolUse]:
    """Extract tool_use blocks from LLM response content.

    Accepts:
    - A list of Anthropic-style content blocks
      ``[{"type": "tool_use", "id": ..., "name": ..., "input": {...}}, ...]``
    - An Anthropic message dict ``{"role": "assistant", "content": [...]}``
    - An OpenAI-style choices[0].message dict with ``tool_calls``
    - A plain string (returns empty list)
    - ``None`` (returns empty list)

    Returns:
        List of ToolUse in the order they appear. Empty if none found.
    """
    if content is None or isinstance(content, str):
        return []

    # Anthropic/OpenAI message dict
    if isinstance(content, dict):
        results: list["ToolUse"] = []
        # OpenAI-style: tool_calls key takes priority
        if "tool_calls" in content and content["tool_calls"]:
            results.extend(_extract_openai(content["tool_calls"]))
        # Anthropic-style: content list (only if tool_calls not present)
        if not results and "content" in content and isinstance(content["content"], list):
            results.extend(extract_tool_uses(content["content"]))
        return results

    if not isinstance(content, list):
        return []

    results: list[ToolUse] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        # Anthropic tool_use block
        if block_type == "tool_use":
            tool_id = block.get("id", "")
            name = block.get("name", "")
            input_data = block.get("input", {})
            if not isinstance(input_data, dict):
                input_data = {}
            results.append(ToolUse(
                id=tool_id,
                name=name,
                input=copy.deepcopy(input_data),
                raw=copy.deepcopy(block),
            ))

        # OpenAI-style tool_calls list embedded in content
        elif block_type == "function" or "function" in block:
            fn = block.get("function", {})
            import json as _json
            raw_args = fn.get("arguments", "{}")
            try:
                args = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            results.append(ToolUse(
                id=block.get("id", ""),
                name=fn.get("name", ""),
                input=args,
                raw=copy.deepcopy(block),
            ))

    return results


def _extract_openai(tool_calls: list[Any]) -> list[ToolUse]:
    """Extract from OpenAI tool_calls array."""
    import json as _json
    results: list[ToolUse] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            args = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}
        results.append(ToolUse(
            id=tc.get("id", ""),
            name=fn.get("name", ""),
            input=args,
            raw=copy.deepcopy(tc),
        ))
    return results


def extract_from_message(message: dict[str, Any]) -> list[ToolUse]:
    """Extract tool_use blocks from a message dict.

    Equivalent to ``extract_tool_uses(message)``.
    """
    return extract_tool_uses(message)


def extract_text(content: Any) -> str:
    """Extract and join all text blocks from Anthropic-style content.

    Useful when a response contains both text and tool_use blocks and you
    want just the text part.

    Args:
        content: content block list, message dict, or plain string.

    Returns:
        Concatenated text from all text blocks, joined by newlines.
        Returns the string unchanged if content is a plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return extract_text(content.get("content", ""))
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)
