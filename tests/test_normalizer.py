"""
Test suite for all dialect parsers: DeepSeek DSML, MiniMax XML, Qwen Text, Generic JSON.
Tests the individual parsers and the unified normalize_tool_calls aggregator.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from server import (
    parse_deepseek_dsml,
    parse_minimax_xml,
    parse_qwen_text,
    parse_generic_json_blocks,
    normalize_tool_calls
)
import json

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


# ======================== DeepSeek DSML Tests ========================
print("\n=== DeepSeek DSML Parser ===")

# Test 1: Full DSML syntax from doc §4.1
dsml_input = '''I'll read the file for you.
<｜DSML｜function_calls>
<｜DSML｜invoke name="read_file">
<｜DSML｜parameter name="path" string="true">/home/user/test.py</｜DSML｜parameter>
<｜DSML｜parameter name="line_count" string="false">100</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>'''

clean, tools = parse_deepseek_dsml(dsml_input)
check("DSML: Extracts tool call", len(tools) == 1)
check("DSML: Correct function name", tools[0]["function"]["name"] == "read_file" if tools else False)
args = json.loads(tools[0]["function"]["arguments"]) if tools else {}
check("DSML: String param preserved", args.get("path") == "/home/user/test.py")
check("DSML: Non-string param cast to int", args.get("line_count") == 100)
check("DSML: Cleaned text preserved", "I'll read the file for you." in clean)
check("DSML: Tags removed from text", "DSML" not in clean)

# Test 2: Simpler <｜tool calls begin｜> variant
simple_input = '<｜tool calls begin｜>\n<｜tool call begin｜>{"name": "get_weather", "arguments": {"location": "London"}}<｜tool call end｜>\n<｜tool calls end｜>'
clean2, tools2 = parse_deepseek_dsml(simple_input)
check("DSML-simple: Extracts tool", len(tools2) == 1)
check("DSML-simple: Correct name", tools2[0]["function"]["name"] == "get_weather" if tools2 else False)


# ======================== MiniMax XML Tests ========================
print("\n=== MiniMax XML Parser ===")

minimax_input = '''Let me check that for you.
<minimax:tool_call>
<invoke name="bash">
<parameter name="command">ls -la /tmp</parameter>
<parameter name="timeout">30</parameter>
<parameter name="verbose">true</parameter>
</invoke>
</minimax:tool_call>'''

clean3, tools3 = parse_minimax_xml(minimax_input)
check("MiniMax: Extracts tool call", len(tools3) == 1)
check("MiniMax: Correct function name", tools3[0]["function"]["name"] == "bash" if tools3 else False)
args3 = json.loads(tools3[0]["function"]["arguments"]) if tools3 else {}
check("MiniMax: String param preserved", args3.get("command") == "ls -la /tmp")
check("MiniMax: Integer param cast", args3.get("timeout") == 30)
check("MiniMax: Boolean param cast", args3.get("verbose") == True)
check("MiniMax: Cleaned text preserved", "Let me check" in clean3)
check("MiniMax: Tags removed", "minimax" not in clean3)

# Test: No MiniMax content returns unchanged
clean_none, tools_none = parse_minimax_xml("Just a normal message")
check("MiniMax: No-op for normal text", clean_none == "Just a normal message" and len(tools_none) == 0)


# ======================== Qwen Text Tests ========================
print("\n=== Qwen Text Parser ===")

qwen_input = '''I need to read a file to understand the project structure.

Tool usage:
Tool: read_file
Arguments: {"path": "/workspace/src/main.py", "encoding": "utf-8"}'''

clean4, tools4 = parse_qwen_text(qwen_input)
check("Qwen: Extracts tool call", len(tools4) == 1)
check("Qwen: Correct function name", tools4[0]["function"]["name"] == "read_file" if tools4 else False)
args4 = json.loads(tools4[0]["function"]["arguments"]) if tools4 else {}
check("Qwen: Path argument correct", args4.get("path") == "/workspace/src/main.py")
check("Qwen: Cleaned text preserved", "project structure" in clean4)
check("Qwen: Tool block removed", "Tool:" not in clean4)

# Test: No Qwen content returns unchanged
clean_nq, tools_nq = parse_qwen_text("Just a normal message")
check("Qwen: No-op for normal text", clean_nq == "Just a normal message" and len(tools_nq) == 0)


# ======================== Generic JSON Block Tests ========================
print("\n=== Generic JSON Block Parser ===")

generic_input = '''I will check the files now.
```json
{
  "name": "read_file",
  "arguments": {
    "path": "/foo/bar.txt"
  }
}
```
Done.'''

clean5, tools5 = parse_generic_json_blocks(generic_input)
check("Generic: Extracts tool call", len(tools5) == 1)
check("Generic: Correct function name", tools5[0]["function"]["name"] == "read_file" if tools5 else False)
check("Generic: Cleaned text has preamble", "I will check" in clean5)
check("Generic: JSON block removed", "```json" not in clean5)


# ======================== Unified Normalizer Tests ========================
print("\n=== Unified normalize_tool_calls ===")

# Test with existing native tool calls + embedded
existing = [{
    "id": "call_native123",
    "type": "function",
    "function": {"name": "native_tool", "arguments": '{"key": "val"}'}
}]
mixed_input = '''Here is my response.
<minimax:tool_call>
<invoke name="bash">
<parameter name="command">echo hello</parameter>
</invoke>
</minimax:tool_call>'''

clean6, tools6 = normalize_tool_calls(mixed_input, existing)
check("Unified: Native + embedded = 2 tools", len(tools6) == 2)
check("Unified: Native tool preserved", any(t["function"]["name"] == "native_tool" for t in tools6))
check("Unified: Embedded tool added", any(t["function"]["name"] == "bash" for t in tools6))

# Test with None existing_tool_calls
clean7, tools7 = normalize_tool_calls("No tools here", None)
check("Unified: None tools returns empty list", isinstance(tools7, list) and len(tools7) == 0)
check("Unified: Text preserved", clean7 == "No tools here")

# Test with empty string
clean8, tools8 = normalize_tool_calls("", [])
check("Unified: Empty string handled", clean8 == "" and tools8 == [])


# ======================== Summary ========================
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
if FAIL > 0:
    print("⚠️  Some tests failed!")
    sys.exit(1)
else:
    print("✅ All tests passed!")
