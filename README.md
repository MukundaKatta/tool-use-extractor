# tool-use-extractor

[![CI](https://github.com/MukundaKatta/tool-use-extractor/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/tool-use-extractor/actions/workflows/ci.yml)

Extract `tool_use` / `tool_calls` blocks from Anthropic and OpenAI LLM
response content with a single, uniform helper.

- **Zero dependencies** — pure standard-library Python.
- **No SDK required** — works directly on the raw `dict` / `list` shapes the
  APIs return, so you don't have to import or pin `anthropic` or `openai`.
- **One result type** — Anthropic `tool_use` blocks and OpenAI `tool_calls`
  both come back as the same `ToolUse` objects.
- Python 3.10+. MIT licensed. Ships a `py.typed` marker for type checkers.

## Install

```bash
pip install tool-use-extractor
```

## Quick start

`extract_tool_uses` accepts whatever shape you have — a content block list, a
full message dict (Anthropic *or* OpenAI), a plain string, or `None` — and
always returns a `list[ToolUse]`.

```python
from tool_use_extractor import extract_tool_uses

# An Anthropic-style assistant response (this is the raw API shape).
response_content = [
    {"type": "text", "text": "Let me look that up."},
    {
        "type": "tool_use",
        "id": "toolu_abc123",
        "name": "web_search",
        "input": {"q": "Python history"},
    },
]

for tool in extract_tool_uses(response_content):
    print(tool.name)   # "web_search"
    print(tool.id)     # "toolu_abc123"
    print(tool.input)  # {"q": "Python history"}
    print(tool.get("q"))  # "Python history"
```

With the real Anthropic SDK you would pass `message.content` straight through:

```python
import anthropic
from tool_use_extractor import extract_tool_uses

client = anthropic.Anthropic()
message = client.messages.create(model="claude-sonnet-4-5", max_tokens=1024, ...)
tools = extract_tool_uses(message.content)  # or extract_tool_uses(message.model_dump())
```

## From a message dict

You can pass an entire assistant message; the extractor figures out the
format. OpenAI `tool_calls` (where `arguments` is a JSON *string*) are parsed
into a real dict for you.

```python
from tool_use_extractor import extract_tool_uses

# Anthropic message dict
extract_tool_uses({"role": "assistant", "content": [...]})

# OpenAI message dict (choices[0].message)
extract_tool_uses({
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
})
# -> [ToolUse(id="call_openai_001", name="get_weather",
#             input={"city": "Paris", "unit": "celsius"})]
```

## Extract just the text

When a response mixes text and tool calls and you only want the prose:

```python
from tool_use_extractor import extract_text

extract_text([
    {"type": "text", "text": "First."},
    {"type": "tool_use", "id": "x", "name": "t", "input": {}},
    {"type": "text", "text": "Second."},
])
# -> "First.\nSecond."
```

## API reference

### `extract_tool_uses(content) -> list[ToolUse]`

Extract every tool call from `content`. Accepted inputs:

| Input                                                   | Behavior                                        |
| ------------------------------------------------------- | ----------------------------------------------- |
| Anthropic content block `list`                          | Returns the `tool_use` blocks in order          |
| Anthropic message `dict` (`{"role", "content": [...]}`) | Recurses into `content`                         |
| OpenAI message `dict` (with `tool_calls`)               | Parses `tool_calls` (JSON-string args → `dict`) |
| Plain `str`                                             | Returns `[]`                                     |
| `None`                                                  | Returns `[]`                                     |

If a message dict has both `tool_calls` and `content`, `tool_calls` wins.
Malformed entries (non-dict blocks, unparseable JSON arguments, non-object
arguments) are skipped or coerced to an empty `input` rather than raising.

### `extract_from_message(message) -> list[ToolUse]`

Convenience alias for `extract_tool_uses(message)`, documenting intent when you
know you are passing a full message dict.

### `extract_text(content) -> str`

Concatenate every `text` block, joined by newlines. A plain string is returned
unchanged; `None`, empty lists, and content with no text blocks return `""`.

### `ToolUse`

A dataclass describing one extracted tool call.

| Attribute / method     | Description                                            |
| ---------------------- | ----------------------------------------------------- |
| `id: str`              | Unique tool-call id (e.g. `"toolu_..."`, `"call_..."`) |
| `name: str`            | Tool name                                             |
| `input: dict`          | Arguments (deep-copied; safe to mutate)               |
| `raw: dict`            | The original block (deep-copied)                      |
| `tool.get(key, default=None)` | Read an argument with an optional default      |
| `key in tool`          | `True` if `key` is present in `input`                 |

The `input` and `raw` dicts are deep copies, so mutating a `ToolUse` never
touches your original response object.

### `ToolUseExtractorError`

Exception subclass reserved for future strict-mode parsing failures. The
current lenient API does not raise it.

## Development

The test suite uses only the standard-library `unittest` framework (no
third-party test dependencies):

```bash
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
