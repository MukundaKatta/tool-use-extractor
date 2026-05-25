# tool-use-extractor

Extract tool_use blocks from Anthropic and OpenAI LLM response content.

Zero dependencies. Python 3.10+. MIT.

## Install

```bash
pip install tool-use-extractor
```

## Usage

```python
from tool_use_extractor import extract_tool_uses, extract_text

# Anthropic content block list
content = response.content  # list of blocks from API response
tools = extract_tool_uses(content)

for t in tools:
    print(t.name)   # "web_search"
    print(t.id)     # "toolu_abc123"
    print(t.input)  # {"q": "Python history"}
```

## From a message dict

```python
# Anthropic message dict
tools = extract_tool_uses({"role": "assistant", "content": [...]})

# OpenAI message dict (with tool_calls)
tools = extract_tool_uses({"role": "assistant", "tool_calls": [...]})
```

## ToolUse helpers

```python
t = tools[0]
t.get("q")           # get an arg with optional default
"q" in t             # check if arg exists
t.input              # full dict
t.raw                # original block dict
```

## Extract text

```python
from tool_use_extractor import extract_text

# Get just the text blocks, skipping tool_use blocks
text = extract_text(response.content)
```

## License

MIT
