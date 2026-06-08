"""Tests for tool-use-extractor.

Uses the Python standard-library ``unittest`` framework only (no third-party
dependencies). Run with::

    python3 -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tool_use_extractor import (  # noqa: E402
    ToolUse,
    ToolUseExtractorError,
    extract_from_message,
    extract_text,
    extract_tool_uses,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ANTHROPIC_CONTENT = [
    {"type": "text", "text": "Let me search for that."},
    {
        "type": "tool_use",
        "id": "toolu_abc123",
        "name": "web_search",
        "input": {"q": "Python history"},
    },
]

ANTHROPIC_CONTENT_MULTI = [
    {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "a"}},
    {"type": "tool_use", "id": "call_2", "name": "read_file", "input": {"path": "/x"}},
]

OPENAI_MESSAGE = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_openai_001",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Paris", "unit": "celsius"}',
            },
        }
    ],
}

ANTHROPIC_MESSAGE = {
    "role": "assistant",
    "content": ANTHROPIC_CONTENT,
}


# ---------------------------------------------------------------------------
# extract_tool_uses — Anthropic content list
# ---------------------------------------------------------------------------

class TestExtractAnthropicContent(unittest.TestCase):
    def test_extract_anthropic_basic(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "web_search")
        self.assertEqual(tools[0].id, "toolu_abc123")
        self.assertEqual(tools[0].input, {"q": "Python history"})

    def test_extract_returns_tooluse(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertIsInstance(tools[0], ToolUse)

    def test_extract_multiple(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT_MULTI)
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0].name, "search")
        self.assertEqual(tools[1].name, "read_file")

    def test_extract_order_preserved(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT_MULTI)
        self.assertEqual(tools[0].id, "call_1")
        self.assertEqual(tools[1].id, "call_2")

    def test_extract_no_tools(self):
        content = [{"type": "text", "text": "No tools here."}]
        self.assertEqual(extract_tool_uses(content), [])

    def test_extract_empty_list(self):
        self.assertEqual(extract_tool_uses([]), [])

    def test_extract_none(self):
        self.assertEqual(extract_tool_uses(None), [])

    def test_extract_plain_string(self):
        self.assertEqual(extract_tool_uses("just text"), [])

    def test_extract_input_deep_copy(self):
        content = [{"type": "tool_use", "id": "x", "name": "t", "input": {"a": 1}}]
        tools = extract_tool_uses(content)
        tools[0].input["a"] = 99
        # Original content unchanged
        self.assertEqual(content[0]["input"]["a"], 1)

    def test_extract_raw_is_deep_copy(self):
        content = [{"type": "tool_use", "id": "x", "name": "t", "input": {"a": 1}}]
        tools = extract_tool_uses(content)
        tools[0].raw["input"]["a"] = 99
        self.assertEqual(content[0]["input"]["a"], 1)

    def test_extract_skips_non_dict_blocks(self):
        content = [
            "not a dict",
            42,
            {"type": "tool_use", "id": "x", "name": "t", "input": {}},
        ]
        tools = extract_tool_uses(content)
        self.assertEqual(len(tools), 1)

    def test_extract_input_not_a_dict_defaults_empty(self):
        content = [{"type": "tool_use", "id": "x", "name": "t", "input": "oops"}]
        tools = extract_tool_uses(content)
        self.assertEqual(tools[0].input, {})

    def test_extract_unknown_block_type_ignored(self):
        content = [{"type": "image", "source": {}}]
        self.assertEqual(extract_tool_uses(content), [])


# ---------------------------------------------------------------------------
# extract_tool_uses — Anthropic message dict
# ---------------------------------------------------------------------------

class TestExtractAnthropicMessage(unittest.TestCase):
    def test_extract_from_anthropic_message(self):
        tools = extract_tool_uses(ANTHROPIC_MESSAGE)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "web_search")

    def test_extract_from_message_dict_no_content(self):
        msg = {"role": "user"}
        self.assertEqual(extract_tool_uses(msg), [])

    def test_extract_empty_dict(self):
        self.assertEqual(extract_tool_uses({}), [])


# ---------------------------------------------------------------------------
# extract_tool_uses — OpenAI format
# ---------------------------------------------------------------------------

class TestExtractOpenAI(unittest.TestCase):
    def test_extract_openai(self):
        tools = extract_tool_uses(OPENAI_MESSAGE)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "get_weather")
        self.assertEqual(tools[0].id, "call_openai_001")
        self.assertEqual(tools[0].input, {"city": "Paris", "unit": "celsius"})

    def test_extract_openai_parses_json_args(self):
        msg = {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "fn", "arguments": '{"x": 42}'},
                }
            ]
        }
        tools = extract_tool_uses(msg)
        self.assertEqual(tools[0].input["x"], 42)

    def test_extract_openai_invalid_json_args(self):
        msg = {
            "tool_calls": [
                {"id": "c1", "function": {"name": "fn", "arguments": "not json"}},
            ]
        }
        tools = extract_tool_uses(msg)
        self.assertEqual(tools[0].input, {})

    def test_extract_openai_dict_args_passthrough(self):
        # Some SDKs hand back already-parsed dict arguments.
        msg = {
            "tool_calls": [
                {"id": "c1", "function": {"name": "fn", "arguments": {"x": 7}}},
            ]
        }
        tools = extract_tool_uses(msg)
        self.assertEqual(tools[0].input, {"x": 7})

    def test_extract_openai_non_dict_json_args_defaults_empty(self):
        # A JSON array is valid JSON but not a valid argument object.
        msg = {
            "tool_calls": [
                {"id": "c1", "function": {"name": "fn", "arguments": "[1, 2, 3]"}},
            ]
        }
        tools = extract_tool_uses(msg)
        self.assertEqual(tools[0].input, {})

    def test_extract_openai_multiple_tool_calls(self):
        msg = {
            "tool_calls": [
                {"id": "a", "function": {"name": "f1", "arguments": "{}"}},
                {"id": "b", "function": {"name": "f2", "arguments": "{}"}},
            ]
        }
        tools = extract_tool_uses(msg)
        self.assertEqual([t.name for t in tools], ["f1", "f2"])

    def test_extract_openai_function_block_in_content_list(self):
        # OpenAI-style function blocks embedded directly in a content list.
        content = [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key": "v"}'},
            }
        ]
        tools = extract_tool_uses(content)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "lookup")
        self.assertEqual(tools[0].input, {"key": "v"})

    def test_extract_openai_skips_non_dict_tool_calls(self):
        msg = {"tool_calls": ["bad", None, {"id": "ok", "function": {"name": "n"}}]}
        tools = extract_tool_uses(msg)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "n")


# ---------------------------------------------------------------------------
# ToolUse attributes / helpers
# ---------------------------------------------------------------------------

class TestToolUse(unittest.TestCase):
    def test_tooluse_id(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertEqual(tools[0].id, "toolu_abc123")

    def test_tooluse_name(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertEqual(tools[0].name, "web_search")

    def test_tooluse_input(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertEqual(tools[0].input["q"], "Python history")

    def test_tooluse_get(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertEqual(tools[0].get("q"), "Python history")
        self.assertEqual(tools[0].get("missing", "default"), "default")

    def test_tooluse_get_missing_returns_none(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertIsNone(tools[0].get("missing"))

    def test_tooluse_contains(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertIn("q", tools[0])
        self.assertNotIn("missing", tools[0])

    def test_tooluse_raw(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        self.assertEqual(tools[0].raw["type"], "tool_use")

    def test_tooluse_repr(self):
        tools = extract_tool_uses(ANTHROPIC_CONTENT)
        r = repr(tools[0])
        self.assertIn("web_search", r)

    def test_tooluse_missing_input_defaults_to_empty(self):
        content = [{"type": "tool_use", "id": "x", "name": "ping"}]
        tools = extract_tool_uses(content)
        self.assertEqual(tools[0].input, {})

    def test_error_type_is_exception_subclass(self):
        self.assertTrue(issubclass(ToolUseExtractorError, Exception))


# ---------------------------------------------------------------------------
# extract_from_message
# ---------------------------------------------------------------------------

class TestExtractFromMessage(unittest.TestCase):
    def test_extract_from_message_anthropic(self):
        tools = extract_from_message(ANTHROPIC_MESSAGE)
        self.assertEqual(len(tools), 1)

    def test_extract_from_message_openai(self):
        tools = extract_from_message(OPENAI_MESSAGE)
        self.assertEqual(len(tools), 1)


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText(unittest.TestCase):
    def test_extract_text_plain_string(self):
        self.assertEqual(extract_text("hello"), "hello")

    def test_extract_text_from_blocks(self):
        content = [
            {"type": "text", "text": "First."},
            {"type": "tool_use", "id": "x", "name": "t", "input": {}},
            {"type": "text", "text": "Second."},
        ]
        text = extract_text(content)
        self.assertIn("First.", text)
        self.assertIn("Second.", text)

    def test_extract_text_from_message(self):
        msg = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
        self.assertEqual(extract_text(msg), "hi")

    def test_extract_text_none(self):
        self.assertEqual(extract_text(None), "")

    def test_extract_text_empty_list(self):
        self.assertEqual(extract_text([]), "")

    def test_extract_text_no_text_blocks(self):
        content = [{"type": "tool_use", "id": "x", "name": "t", "input": {}}]
        self.assertEqual(extract_text(content), "")

    def test_extract_text_joins_with_newline(self):
        content = [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]
        result = extract_text(content)
        self.assertEqual(result, "Line 1\nLine 2")

    def test_extract_text_skips_empty_text(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "kept"},
        ]
        self.assertEqual(extract_text(content), "kept")

    def test_extract_text_string_message_content(self):
        # Some providers put a plain string in the content field.
        msg = {"role": "assistant", "content": "plain"}
        self.assertEqual(extract_text(msg), "plain")


if __name__ == "__main__":
    unittest.main()
