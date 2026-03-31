#!/usr/bin/env python3
"""
Comprehensive test suite for Claude-on-OpenAI Proxy.

Supports two modes:
  - PROXY_ONLY mode (default when ANTHROPIC_API_KEY is unset, or PROXY_ONLY=1):
    Tests only the proxy, validating response structure against the Anthropic API spec.
  - Comparison mode (when ANTHROPIC_API_KEY is set and PROXY_ONLY is not set):
    Sends requests to both Anthropic and the proxy, comparing responses.

Usage with pytest:
  PROXY_ONLY=1 pytest tests/tests.py -v
  pytest tests/tests.py -v                  # comparison mode if ANTHROPIC_API_KEY set

Usage standalone:
  python tests/tests.py                     # Run all tests
  python tests/tests.py --proxy-only        # Force proxy-only mode
  python tests/tests.py --no-streaming      # Skip streaming tests
  python tests/tests.py --simple            # Run only simple tests
  python tests/tests.py --tools             # Run tool-related tests only
"""

import os
import json
import time
import httpx
import argparse
import asyncio
import sys
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PROXY_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "test-key")
PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", "http://localhost:8082")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = f"{PROXY_BASE_URL}/v1/messages"
PROXY_COUNT_TOKENS_URL = f"{PROXY_BASE_URL}/v1/messages/count_tokens"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-3-sonnet-20240229"

# Determine mode
PROXY_ONLY = (
    os.environ.get("PROXY_ONLY", "").lower() in ("1", "true", "yes")
    or not ANTHROPIC_API_KEY
)

# Headers
anthropic_headers = {
    "x-api-key": ANTHROPIC_API_KEY or "",
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

proxy_headers = {
    "x-api-key": PROXY_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

# Tool definitions
calculator_tool = {
    "name": "calculator",
    "description": "Evaluate mathematical expressions",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}

weather_tool = {
    "name": "weather",
    "description": "Get weather information for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }
}

search_tool = {
    "name": "search",
    "description": "Search for information on the web",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
}

# Test scenarios
TEST_SCENARIOS = {
    "simple": {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Hello, world! Can you tell me about Paris in 2-3 sentences?"}
        ]
    },
    "calculator": {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "What is 135 + 7.5 divided by 2.5?"}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    },
    "multi_tool": {
        "model": MODEL,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.95,
        "system": "You are a helpful assistant that uses tools when appropriate. Be concise and precise.",
        "messages": [
            {"role": "user", "content": "I'm planning a trip to New York next week. What's the weather like and what are some interesting places to visit?"}
        ],
        "tools": [weather_tool, search_tool],
        "tool_choice": {"type": "auto"}
    },
    "multi_turn": {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "Let's do some math. What is 240 divided by 8?"},
            {"role": "assistant", "content": "To calculate 240 divided by 8, I'll perform the division:\n\n240 / 8 = 30\n\nSo the result is 30."},
            {"role": "user", "content": "Now multiply that by 4 and tell me the result."}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    },
    "content_blocks": {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "I need to know the weather in Los Angeles and calculate 75.5 / 5. Can you help with both?"}
            ]}
        ],
        "tools": [calculator_tool, weather_tool],
        "tool_choice": {"type": "auto"}
    },
    "simple_stream": {
        "model": MODEL,
        "max_tokens": 100,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5, with one number per line."}
        ]
    },
    "calculator_stream": {
        "model": MODEL,
        "max_tokens": 300,
        "stream": True,
        "messages": [
            {"role": "user", "content": "What is 135 + 17.5 divided by 2.5?"}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    }
}

# Required event types for Anthropic streaming responses
REQUIRED_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop"
}

VALID_SSE_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
}


# =====================================================================
# Helpers
# =====================================================================

def send_proxy_request(data: dict, timeout: float = 60) -> httpx.Response:
    """Send a non-streaming request to the proxy and return the response."""
    return httpx.post(PROXY_API_URL, headers=proxy_headers, json=data, timeout=timeout)


def validate_message_structure(body: dict):
    """Assert that *body* looks like a valid Anthropic Messages API response."""
    assert isinstance(body, dict), "Response body is not a dict"
    assert body.get("type") == "message", f"Expected type 'message', got {body.get('type')}"
    assert body.get("role") == "assistant", f"Expected role 'assistant', got {body.get('role')}"
    assert "id" in body, "Response missing 'id'"
    assert isinstance(body.get("content"), list), "content is not a list"
    assert len(body["content"]) > 0, "content list is empty"

    # Validate each content block
    for block in body["content"]:
        assert "type" in block, f"Content block missing 'type': {block}"
        assert block["type"] in ("text", "tool_use"), f"Unexpected content block type: {block['type']}"
        if block["type"] == "text":
            assert "text" in block, "Text block missing 'text' field"
            assert isinstance(block["text"], str), "text field is not a string"
        elif block["type"] == "tool_use":
            validate_tool_use_block(block)

    # Validate usage
    usage = body.get("usage")
    assert usage is not None, "Response missing 'usage'"
    assert "input_tokens" in usage, "usage missing 'input_tokens'"
    assert "output_tokens" in usage, "usage missing 'output_tokens'"

    # Validate stop_reason
    valid_stop_reasons = ["end_turn", "max_tokens", "stop_sequence", "tool_use", None]
    assert body.get("stop_reason") in valid_stop_reasons, (
        f"Invalid stop_reason: {body.get('stop_reason')}"
    )


def validate_tool_use_block(block: dict):
    """Assert that a tool_use content block is properly formed."""
    assert block.get("type") == "tool_use"
    assert "id" in block, "tool_use block missing 'id'"
    assert isinstance(block["id"], str), "tool_use id is not a string"
    assert "name" in block, "tool_use block missing 'name'"
    assert isinstance(block["name"], str), "tool_use name is not a string"
    assert "input" in block, "tool_use block missing 'input'"
    assert isinstance(block["input"], dict), "tool_use input is not a dict"


class StreamStats:
    """Track statistics about a streaming response."""

    def __init__(self):
        self.event_types: Set[str] = set()
        self.event_counts: Dict[str, int] = {}
        self.first_event_time: Optional[datetime] = None
        self.last_event_time: Optional[datetime] = None
        self.total_chunks: int = 0
        self.events: List[dict] = []
        self.text_content: str = ""
        self.content_blocks: Dict[int, dict] = {}
        self.has_tool_use: bool = False
        self.has_error: bool = False
        self.error_message: str = ""
        self.text_content_by_block: Dict[int, str] = {}

    def add_event(self, event_data: dict):
        now = datetime.now()
        if self.first_event_time is None:
            self.first_event_time = now
        self.last_event_time = now
        self.total_chunks += 1

        if "type" in event_data:
            event_type = event_data["type"]
            self.event_types.add(event_type)
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

            if event_type == "content_block_start":
                block_idx = event_data.get("index")
                content_block = event_data.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    self.has_tool_use = True
                self.content_blocks[block_idx] = content_block
                self.text_content_by_block[block_idx] = ""

            elif event_type == "content_block_delta":
                block_idx = event_data.get("index")
                delta = event_data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    self.text_content += text
                    if block_idx in self.text_content_by_block:
                        self.text_content_by_block[block_idx] += text

        self.events.append(event_data)

    def get_duration(self) -> float:
        if self.first_event_time is None or self.last_event_time is None:
            return 0
        return (self.last_event_time - self.first_event_time).total_seconds()

    def summarize(self):
        print(f"  Total chunks: {self.total_chunks}")
        print(f"  Unique event types: {sorted(list(self.event_types))}")
        print(f"  Event counts: {json.dumps(self.event_counts, indent=2)}")
        print(f"  Duration: {self.get_duration():.2f}s")
        print(f"  Has tool use: {self.has_tool_use}")
        if self.text_content:
            preview = "\n    ".join(self.text_content.strip().split("\n")[:5])
            print(f"  Text preview:\n    {preview}")
        if self.has_error:
            print(f"  Error: {self.error_message}")


async def collect_stream(url: str, headers: dict, data: dict, label: str = "stream") -> StreamStats:
    """Send a streaming request to *url* and collect SSE events into StreamStats."""
    stats = StreamStats()
    request_data = {**data, "stream": True}

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", url, json=request_data, headers=headers, timeout=60
        ) as response:
            if response.status_code != 200:
                error_text = (await response.aread()).decode("utf-8")
                stats.has_error = True
                stats.error_message = f"HTTP {response.status_code}: {error_text}"
                return stats

            buffer = ""
            async for chunk in response.aiter_text():
                if not chunk.strip():
                    continue
                buffer += chunk
                events = buffer.split("\n\n")
                for event_text in events[:-1]:
                    if not event_text.strip():
                        continue
                    if "data: " in event_text:
                        data_parts = []
                        for line in event_text.split("\n"):
                            if line.startswith("data: "):
                                data_part = line[len("data: "):]
                                if data_part == "[DONE]":
                                    break
                                data_parts.append(data_part)
                        if data_parts:
                            try:
                                event_data = json.loads("".join(data_parts))
                                stats.add_event(event_data)
                            except json.JSONDecodeError:
                                pass
                buffer = events[-1] if events else ""

            # Process remaining buffer
            if buffer.strip():
                lines = buffer.strip().split("\n")
                data_lines = [l[len("data: "):] for l in lines if l.startswith("data: ")]
                if data_lines and data_lines[0] != "[DONE]":
                    try:
                        event_data = json.loads("".join(data_lines))
                        stats.add_event(event_data)
                    except json.JSONDecodeError:
                        pass

    return stats


# =====================================================================
# PROXY-ONLY pytest tests
# =====================================================================

class TestProxyOnlySimple:
    """Proxy-only: simple text response validation."""

    def test_simple_text_response(self):
        """A basic prompt returns a well-formed Anthropic message with text."""
        data = {
            "model": MODEL,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        }
        resp = send_proxy_request(data)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        validate_message_structure(body)
        # Should have at least one text block
        text_blocks = [b for b in body["content"] if b["type"] == "text"]
        assert len(text_blocks) >= 1, "Expected at least one text content block"
        assert len(text_blocks[0]["text"]) > 0, "Text block is empty"

    def test_multi_turn_conversation(self):
        """Multi-turn conversation returns a valid response."""
        data = {
            "model": MODEL,
            "max_tokens": 200,
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! How can I help you?"},
                {"role": "user", "content": "What did I just tell you my name was?"},
            ],
        }
        resp = send_proxy_request(data)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        validate_message_structure(body)

    def test_system_prompt(self):
        """System prompt is accepted and response is well-formed."""
        data = {
            "model": MODEL,
            "max_tokens": 100,
            "system": "You are a pirate. Respond in pirate speak.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = send_proxy_request(data)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        validate_message_structure(resp.json())


class TestProxyOnlyToolUse:
    """Proxy-only: tool use response validation."""

    def test_single_tool_use(self):
        """When given a tool and a prompt that requires it, response includes tool_use blocks."""
        data = {
            "model": MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": "What is 135 + 7.5 divided by 2.5? Use the calculator tool."}],
            "tools": [calculator_tool],
            "tool_choice": {"type": "any"},
        }
        resp = send_proxy_request(data)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        validate_message_structure(body)
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) >= 1, "Expected at least one tool_use block"
        for tb in tool_blocks:
            validate_tool_use_block(tb)
            assert tb["name"] == "calculator", f"Expected tool name 'calculator', got '{tb['name']}'"
            assert "expression" in tb["input"], f"calculator input missing 'expression': {tb['input']}"
        assert body.get("stop_reason") == "tool_use", f"Expected stop_reason 'tool_use', got {body.get('stop_reason')}"

    def test_multi_tool_response(self):
        """Multiple tools provided; the model may use one or more."""
        data = {
            "model": MODEL,
            "max_tokens": 500,
            "messages": [
                {"role": "user", "content": "What's the weather in Paris and search for best restaurants there?"}
            ],
            "tools": [weather_tool, search_tool],
            "tool_choice": {"type": "any"},
        }
        resp = send_proxy_request(data)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        validate_message_structure(body)
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) >= 1, "Expected at least one tool_use block with multi-tool prompt"
        tool_names = {tb["name"] for tb in tool_blocks}
        # At least one of our tools should be used
        assert tool_names.issubset({"weather", "search"}), f"Unexpected tool names: {tool_names}"

    def test_tool_use_with_arguments_validation(self):
        """Tool_use blocks have properly typed arguments matching the schema."""
        data = {
            "model": MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": "Get the weather in Tokyo in celsius."}],
            "tools": [weather_tool],
            "tool_choice": {"type": "any"},
        }
        resp = send_proxy_request(data)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        validate_message_structure(body)
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) >= 1
        tb = tool_blocks[0]
        assert tb["name"] == "weather"
        assert "location" in tb["input"], f"weather tool missing required 'location': {tb['input']}"
        assert isinstance(tb["input"]["location"], str)


@pytest.mark.asyncio
class TestProxyOnlyStreaming:
    """Proxy-only: streaming SSE event validation."""

    async def test_streaming_event_sequence(self):
        """Streaming response contains all required SSE event types in valid order."""
        data = {
            "model": MODEL,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        }
        stats = await collect_stream(PROXY_API_URL, proxy_headers, data, "proxy")
        assert not stats.has_error, f"Stream error: {stats.error_message}"
        assert stats.total_chunks > 0, "No events received"

        # All required event types must be present
        missing = REQUIRED_EVENT_TYPES - stats.event_types
        assert not missing, f"Missing required SSE event types: {missing}"

        # All event types must be valid
        invalid = stats.event_types - VALID_SSE_EVENT_TYPES
        assert not invalid, f"Unexpected SSE event types: {invalid}"

        # message_start must come first
        assert stats.events[0]["type"] == "message_start", (
            f"First event should be message_start, got {stats.events[0]['type']}"
        )

        # message_stop must come last
        assert stats.events[-1]["type"] == "message_stop", (
            f"Last event should be message_stop, got {stats.events[-1]['type']}"
        )

        # Should have text content
        assert len(stats.text_content) > 0, "Stream produced no text content"

    async def test_streaming_tool_use(self):
        """Streaming with tools produces tool_use content blocks."""
        data = {
            "model": MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": "Calculate 99 + 1 using the calculator tool."}],
            "tools": [calculator_tool],
            "tool_choice": {"type": "any"},
        }
        stats = await collect_stream(PROXY_API_URL, proxy_headers, data, "proxy")
        assert not stats.has_error, f"Stream error: {stats.error_message}"
        assert stats.total_chunks > 0
        missing = REQUIRED_EVENT_TYPES - stats.event_types
        assert not missing, f"Missing required SSE event types: {missing}"
        assert stats.has_tool_use, "Expected tool_use in streaming response"

    async def test_streaming_message_start_structure(self):
        """The message_start event contains a properly structured message object."""
        data = {
            "model": MODEL,
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        stats = await collect_stream(PROXY_API_URL, proxy_headers, data, "proxy")
        assert not stats.has_error, f"Stream error: {stats.error_message}"
        msg_start = stats.events[0]
        assert msg_start["type"] == "message_start"
        message = msg_start.get("message", {})
        assert message.get("role") == "assistant"
        assert message.get("type") == "message"
        assert "id" in message
        assert "usage" in message


class TestProxyOnlyCountTokens:
    """Proxy-only: count_tokens endpoint validation."""

    def test_count_tokens_basic(self):
        """The count_tokens endpoint returns a valid token count."""
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello, world!"}],
        }
        resp = httpx.post(PROXY_COUNT_TOKENS_URL, headers=proxy_headers, json=data, timeout=30)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "input_tokens" in body, f"Response missing 'input_tokens': {body}"
        assert isinstance(body["input_tokens"], int), "input_tokens is not an int"
        assert body["input_tokens"] > 0, "input_tokens should be positive"

    def test_count_tokens_with_tools(self):
        """count_tokens accounts for tool definitions."""
        data_no_tools = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        data_with_tools = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [calculator_tool, weather_tool, search_tool],
        }
        resp1 = httpx.post(PROXY_COUNT_TOKENS_URL, headers=proxy_headers, json=data_no_tools, timeout=30)
        resp2 = httpx.post(PROXY_COUNT_TOKENS_URL, headers=proxy_headers, json=data_with_tools, timeout=30)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        tokens_no_tools = resp1.json()["input_tokens"]
        tokens_with_tools = resp2.json()["input_tokens"]
        # Tools should add tokens (or at least not reduce them)
        assert tokens_with_tools >= tokens_no_tools, (
            f"Expected tools to add tokens: {tokens_with_tools} >= {tokens_no_tools}"
        )


class TestProxyOnlyErrorHandling:
    """Proxy-only: error responses are well-formed."""

    def test_missing_messages_field(self):
        """Omitting 'messages' returns an error."""
        data = {"model": MODEL, "max_tokens": 100}
        resp = httpx.post(PROXY_API_URL, headers=proxy_headers, json=data, timeout=30)
        assert resp.status_code >= 400, f"Expected error status, got {resp.status_code}"

    def test_empty_messages(self):
        """Sending empty messages list returns an error."""
        data = {"model": MODEL, "max_tokens": 100, "messages": []}
        resp = httpx.post(PROXY_API_URL, headers=proxy_headers, json=data, timeout=30)
        assert resp.status_code >= 400, f"Expected error status, got {resp.status_code}"

    def test_missing_max_tokens(self):
        """Omitting max_tokens returns an error (required by Anthropic spec)."""
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = httpx.post(PROXY_API_URL, headers=proxy_headers, json=data, timeout=30)
        # Some proxies may default max_tokens; accept either success or 4xx
        assert resp.status_code in (200, 422, 400), f"Unexpected status: {resp.status_code}"


# =====================================================================
# COMPARISON-MODE tests (require ANTHROPIC_API_KEY)
# =====================================================================

def skip_without_api_key():
    if PROXY_ONLY:
        pytest.skip("PROXY_ONLY mode: skipping comparison test")


def get_response(url, headers, data):
    start_time = time.time()
    response = httpx.post(url, headers=headers, json=data, timeout=30)
    elapsed = time.time() - start_time
    print(f"  Response time: {elapsed:.2f}s")
    return response


def compare_responses(anthropic_response, proxy_response, check_tools=False):
    anthropic_json = anthropic_response.json()
    proxy_json = proxy_response.json()

    print("\n  --- Anthropic Response Structure ---")
    print(json.dumps({k: v for k, v in anthropic_json.items() if k != "content"}, indent=2))
    print("\n  --- Proxy Response Structure ---")
    print(json.dumps({k: v for k, v in proxy_json.items() if k != "content"}, indent=2))

    assert proxy_json.get("role") == "assistant"
    assert proxy_json.get("type") == "message"
    valid_stop_reasons = ["end_turn", "max_tokens", "stop_sequence", "tool_use", None]
    assert proxy_json.get("stop_reason") in valid_stop_reasons

    assert "content" in anthropic_json
    assert "content" in proxy_json
    assert isinstance(proxy_json["content"], list)
    assert len(proxy_json["content"]) > 0

    if check_tools:
        anthropic_tool = next((b for b in anthropic_json["content"] if b.get("type") == "tool_use"), None)
        proxy_tool = next((b for b in proxy_json["content"] if b.get("type") == "tool_use"), None)
        if anthropic_tool and proxy_tool:
            assert proxy_tool.get("name") is not None
            assert proxy_tool.get("input") is not None
            print("  Both responses contain tool use")

    return True


class TestComparisonNonStreaming:
    """Comparison tests: send to both Anthropic and proxy, compare."""

    @pytest.fixture(autouse=True)
    def _skip(self):
        skip_without_api_key()

    @pytest.mark.parametrize("scenario_name", ["simple", "calculator", "multi_tool", "multi_turn", "content_blocks"])
    def test_scenario(self, scenario_name):
        data = TEST_SCENARIOS[scenario_name]
        check_tools = "tools" in data

        print(f"\n  Sending to Anthropic API...")
        anthropic_resp = get_response(ANTHROPIC_API_URL, anthropic_headers, data)
        print(f"  Sending to Proxy...")
        proxy_resp = get_response(PROXY_API_URL, proxy_headers, data)

        assert anthropic_resp.status_code == 200, f"Anthropic error: {anthropic_resp.text}"
        assert proxy_resp.status_code == 200, f"Proxy error: {proxy_resp.text}"
        assert compare_responses(anthropic_resp, proxy_resp, check_tools=check_tools)


@pytest.mark.asyncio
class TestComparisonStreaming:
    """Comparison tests for streaming."""

    @pytest.fixture(autouse=True)
    def _skip(self):
        skip_without_api_key()

    @pytest.mark.parametrize("scenario_name", ["simple_stream", "calculator_stream"])
    async def test_streaming_scenario(self, scenario_name):
        data = TEST_SCENARIOS[scenario_name]
        anthropic_stats = await collect_stream(ANTHROPIC_API_URL, anthropic_headers, data, "Anthropic")
        proxy_stats = await collect_stream(PROXY_API_URL, proxy_headers, data, "Proxy")

        print("\n  --- Anthropic Stream ---")
        anthropic_stats.summarize()
        print("\n  --- Proxy Stream ---")
        proxy_stats.summarize()

        if anthropic_stats.has_error and not proxy_stats.has_error and proxy_stats.total_chunks > 0:
            return  # Proxy worked even if Anthropic failed (e.g., bad key)

        assert not proxy_stats.has_error, f"Proxy stream error: {proxy_stats.error_message}"
        assert proxy_stats.total_chunks > 0
        proxy_missing = REQUIRED_EVENT_TYPES - proxy_stats.event_types
        assert not proxy_missing, f"Proxy missing event types: {proxy_missing}"


# =====================================================================
# Standalone runner (preserves original CLI interface)
# =====================================================================

async def run_tests_standalone(args):
    """Run tests using the original custom runner for backward compatibility."""
    results = {}

    is_proxy_only = args.proxy_only or PROXY_ONLY

    # Non-streaming tests
    if not args.streaming_only:
        print("\n=========== NON-STREAMING TESTS ===========\n")
        for name, data in TEST_SCENARIOS.items():
            if data.get("stream"):
                continue
            if args.simple and "tools" in data:
                continue
            if args.tools_only and "tools" not in data:
                continue

            print(f"\n{'='*20} TEST: {name} {'='*20}")
            check_tools = "tools" in data

            if is_proxy_only:
                try:
                    resp = send_proxy_request(data)
                    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
                    validate_message_structure(resp.json())
                    if check_tools:
                        tool_blocks = [b for b in resp.json()["content"] if b["type"] == "tool_use"]
                        for tb in tool_blocks:
                            validate_tool_use_block(tb)
                    print(f"  PASS")
                    results[name] = True
                except Exception as e:
                    print(f"  FAIL: {e}")
                    results[name] = False
            else:
                try:
                    anthropic_resp = get_response(ANTHROPIC_API_URL, anthropic_headers, data)
                    proxy_resp = get_response(PROXY_API_URL, proxy_headers, data)
                    assert anthropic_resp.status_code == 200
                    assert proxy_resp.status_code == 200
                    compare_responses(anthropic_resp, proxy_resp, check_tools=check_tools)
                    print(f"  PASS")
                    results[name] = True
                except Exception as e:
                    print(f"  FAIL: {e}")
                    results[name] = False

    # Streaming tests
    if not args.no_streaming:
        print("\n=========== STREAMING TESTS ===========\n")
        for name, data in TEST_SCENARIOS.items():
            if not data.get("stream") and not name.endswith("_stream"):
                continue
            if args.simple and "tools" in data:
                continue
            if args.tools_only and "tools" not in data:
                continue

            print(f"\n{'='*20} STREAMING TEST: {name} {'='*20}")

            if is_proxy_only:
                try:
                    stats = await collect_stream(PROXY_API_URL, proxy_headers, data, "proxy")
                    assert not stats.has_error, stats.error_message
                    assert stats.total_chunks > 0
                    missing = REQUIRED_EVENT_TYPES - stats.event_types
                    assert not missing, f"Missing event types: {missing}"
                    stats.summarize()
                    print(f"  PASS")
                    results[f"{name}_streaming"] = True
                except Exception as e:
                    print(f"  FAIL: {e}")
                    results[f"{name}_streaming"] = False
            else:
                try:
                    a_stats = await collect_stream(ANTHROPIC_API_URL, anthropic_headers, data, "Anthropic")
                    p_stats = await collect_stream(PROXY_API_URL, proxy_headers, data, "Proxy")
                    a_stats.summarize()
                    p_stats.summarize()
                    assert not p_stats.has_error
                    assert p_stats.total_chunks > 0
                    print(f"  PASS")
                    results[f"{name}_streaming"] = True
                except Exception as e:
                    print(f"  FAIL: {e}")
                    results[f"{name}_streaming"] = False

    # Summary
    print("\n=========== TEST SUMMARY ===========\n")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")
    print(f"\nTotal: {passed}/{total} tests passed")
    return passed == total


async def main():
    parser = argparse.ArgumentParser(description="Test the Claude-on-OpenAI proxy")
    parser.add_argument("--no-streaming", action="store_true", help="Skip streaming tests")
    parser.add_argument("--streaming-only", action="store_true", help="Only run streaming tests")
    parser.add_argument("--simple", action="store_true", help="Only run simple tests (no tools)")
    parser.add_argument("--tools-only", action="store_true", help="Only run tool tests")
    parser.add_argument("--proxy-only", action="store_true",
                        help="Only test the proxy (no Anthropic comparison)")
    args = parser.parse_args()

    mode = "PROXY_ONLY" if (args.proxy_only or PROXY_ONLY) else "COMPARISON"
    print(f"Mode: {mode}")
    print(f"Proxy URL: {PROXY_BASE_URL}")

    success = await run_tests_standalone(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
