"""Tests for tool-use-extractor."""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from tool_use_extractor import (
    ToolUse,
    ToolUseExtractorError,
    extract_tool_uses,
    extract_from_message,
    extract_text,
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

def test_extract_anthropic_basic():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert len(tools) == 1
    assert tools[0].name == "web_search"
    assert tools[0].id == "toolu_abc123"
    assert tools[0].input == {"q": "Python history"}

def test_extract_returns_tooluse():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert isinstance(tools[0], ToolUse)

def test_extract_multiple():
    tools = extract_tool_uses(ANTHROPIC_CONTENT_MULTI)
    assert len(tools) == 2
    assert tools[0].name == "search"
    assert tools[1].name == "read_file"

def test_extract_order_preserved():
    tools = extract_tool_uses(ANTHROPIC_CONTENT_MULTI)
    assert tools[0].id == "call_1"
    assert tools[1].id == "call_2"

def test_extract_no_tools():
    content = [{"type": "text", "text": "No tools here."}]
    assert extract_tool_uses(content) == []

def test_extract_empty_list():
    assert extract_tool_uses([]) == []

def test_extract_none():
    assert extract_tool_uses(None) == []

def test_extract_plain_string():
    assert extract_tool_uses("just text") == []

def test_extract_input_deep_copy():
    content = [{"type": "tool_use", "id": "x", "name": "t", "input": {"a": 1}}]
    tools = extract_tool_uses(content)
    tools[0].input["a"] = 99
    # Original content unchanged
    assert content[0]["input"]["a"] == 1


# ---------------------------------------------------------------------------
# extract_tool_uses — Anthropic message dict
# ---------------------------------------------------------------------------

def test_extract_from_anthropic_message():
    tools = extract_tool_uses(ANTHROPIC_MESSAGE)
    assert len(tools) == 1
    assert tools[0].name == "web_search"

def test_extract_from_message_dict_no_content():
    msg = {"role": "user"}
    assert extract_tool_uses(msg) == []


# ---------------------------------------------------------------------------
# extract_tool_uses — OpenAI format
# ---------------------------------------------------------------------------

def test_extract_openai():
    tools = extract_tool_uses(OPENAI_MESSAGE)
    assert len(tools) == 1
    assert tools[0].name == "get_weather"
    assert tools[0].id == "call_openai_001"
    assert tools[0].input == {"city": "Paris", "unit": "celsius"}

def test_extract_openai_parses_json_args():
    msg = {
        "tool_calls": [
            {
                "id": "c1",
                "function": {"name": "fn", "arguments": '{"x": 42}'},
            }
        ]
    }
    tools = extract_tool_uses(msg)
    assert tools[0].input["x"] == 42

def test_extract_openai_invalid_json_args():
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "fn", "arguments": "not json"}},
        ]
    }
    tools = extract_tool_uses(msg)
    assert tools[0].input == {}


# ---------------------------------------------------------------------------
# ToolUse attributes
# ---------------------------------------------------------------------------

def test_tooluse_id():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert tools[0].id == "toolu_abc123"

def test_tooluse_name():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert tools[0].name == "web_search"

def test_tooluse_input():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert tools[0].input["q"] == "Python history"

def test_tooluse_get():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert tools[0].get("q") == "Python history"
    assert tools[0].get("missing", "default") == "default"

def test_tooluse_contains():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert "q" in tools[0]
    assert "missing" not in tools[0]

def test_tooluse_raw():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    assert tools[0].raw["type"] == "tool_use"

def test_tooluse_repr():
    tools = extract_tool_uses(ANTHROPIC_CONTENT)
    r = repr(tools[0])
    assert "web_search" in r

def test_tooluse_missing_input_defaults_to_empty():
    content = [{"type": "tool_use", "id": "x", "name": "ping"}]
    tools = extract_tool_uses(content)
    assert tools[0].input == {}


# ---------------------------------------------------------------------------
# extract_from_message
# ---------------------------------------------------------------------------

def test_extract_from_message_anthropic():
    tools = extract_from_message(ANTHROPIC_MESSAGE)
    assert len(tools) == 1

def test_extract_from_message_openai():
    tools = extract_from_message(OPENAI_MESSAGE)
    assert len(tools) == 1


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

def test_extract_text_plain_string():
    assert extract_text("hello") == "hello"

def test_extract_text_from_blocks():
    content = [
        {"type": "text", "text": "First."},
        {"type": "tool_use", "id": "x", "name": "t", "input": {}},
        {"type": "text", "text": "Second."},
    ]
    text = extract_text(content)
    assert "First." in text
    assert "Second." in text

def test_extract_text_from_message():
    msg = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    assert extract_text(msg) == "hi"

def test_extract_text_none():
    assert extract_text(None) == ""

def test_extract_text_empty_list():
    assert extract_text([]) == ""

def test_extract_text_no_text_blocks():
    content = [{"type": "tool_use", "id": "x", "name": "t", "input": {}}]
    assert extract_text(content) == ""

def test_extract_text_joins_with_newline():
    content = [
        {"type": "text", "text": "Line 1"},
        {"type": "text", "text": "Line 2"},
    ]
    result = extract_text(content)
    assert result == "Line 1\nLine 2"
