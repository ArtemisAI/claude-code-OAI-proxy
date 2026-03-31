<p align="center">
  <h1 align="center">Claude Code OAI Proxy</h1>
  <p align="center">
    Use <a href="https://docs.anthropic.com/en/docs/claude-code">Claude Code</a> with <strong>any model</strong> — DeepSeek, Qwen, GLM, Kimi, MiniMax, Gemini, GPT, or anything behind an OpenAI-compatible API.
  </p>
</p>

<br>

```
                          ┌─────────────────────┐
                          │    Claude Code CLI   │
                          │  (Anthropic Messages │
                          │        API)          │
                          └────────┬─────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │                     │
                          │   Claude Code OAI   │
                          │       Proxy         │
                          │                     │
                          │  • Format translate  │
                          │  • Tool normalize   │
                          │  • Stream transform │
                          │                     │
                          └────────┬────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌───────────┐ ┌───────────┐ ┌───────────┐
             │   vLLM    │ │  Ollama   │ │  OneAPI   │
             │  DeepSeek │ │   Qwen    │ │   GLM     │
             │   Gemini  │ │   Kimi    │ │  MiniMax  │
             └───────────┘ └───────────┘ └───────────┘
```

<br>

## The Problem

Claude Code speaks only the **Anthropic Messages API**. Your models speak **OpenAI Chat Completions**. They can't talk to each other.

Worse — different models emit tool calls in **completely different formats**:

```
DeepSeek    <|DSML|invoke name="Bash">{"command":"ls"}</invoke>
MiniMax     <minimax:tool_call><invoke name="Bash">...</invoke>
GLM         [Bash]\ncommand: ls
Qwen        Tool: Bash\nArguments: {"command":"ls"}
Hermes      <tool_call>{"name":"Bash",...}</tool_call>
```

Some models even **switch formats mid-conversation** or emit malformed hybrids.

This proxy handles all of it — translating requests, normalizing tool calls from 7+ formats, repairing broken JSON, and streaming everything back as clean Anthropic SSE events.

---

## Quick Start

**1. Clone & configure**
```bash
git clone https://github.com/ArtemisAI/Claude-Code-OAI-Proxy.git
cd Claude-Code-OAI-Proxy
cp .env.example .env
```

Edit `.env` with your backend details:
```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.com/v1
BIG_MODEL=your-large-model
SMALL_MODEL=your-small-model
```

**2. Run**
```bash
# Development
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

# Docker (build)
docker build -t claude-oai-proxy .
docker run -d --env-file .env -p 8082:8082 --restart unless-stopped claude-oai-proxy

# Docker (pre-built, amd64 + arm64)
docker run -d --env-file .env -p 8082:8082 --restart unless-stopped \
  ghcr.io/artemisai/claude-code-oai-proxy:main
```

**3. Connect Claude Code**
```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

> `ANTHROPIC_BASE_URL` must **not** include `/v1` — Claude Code appends it automatically.

That's it. Claude Code now talks to your models.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OPENAI_API_KEY` | *(required)* | API key for your backend |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Your API endpoint |
| `BIG_MODEL` | `gpt-4.1` | Model for `claude-sonnet` / `claude-opus` requests |
| `SMALL_MODEL` | `gpt-4.1-mini` | Model for `claude-haiku` requests |
| `OPUS_MODEL` | same as `BIG_MODEL` | Override for `claude-opus` specifically |
| `PREFERRED_PROVIDER` | `openai` | Provider routing: `openai`, `google`, or `anthropic` |

### Model Mapping

```
┌──────────────────────┐         ┌──────────────────────┐
│ Claude Code requests │         │  Your models         │
├──────────────────────┤   ───►  ├──────────────────────┤
│ claude-haiku-*       │         │ SMALL_MODEL          │
│ claude-sonnet-*      │         │ BIG_MODEL            │
│ claude-opus-*        │         │ OPUS_MODEL           │
│ (anything else)      │         │ (passed through)     │
└──────────────────────┘         └──────────────────────┘
```

### Provider Examples

<details>
<summary><b>OpenAI-compatible API</b> — vLLM, Ollama, OneAPI, LiteLLM, etc.</summary>

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-endpoint.com/v1
BIG_MODEL=your-large-model
SMALL_MODEL=your-small-model
```
</details>

<details>
<summary><b>Google Gemini</b> — AI Studio</summary>

```env
PREFERRED_PROVIDER=google
GEMINI_API_KEY=your-key
BIG_MODEL=gemini-2.5-pro
SMALL_MODEL=gemini-2.5-flash
```
</details>

<details>
<summary><b>Google Vertex AI</b> — Application Default Credentials</summary>

```env
PREFERRED_PROVIDER=google
USE_VERTEX_AUTH=true
VERTEX_PROJECT=my-project
VERTEX_LOCATION=us-central1
BIG_MODEL=gemini-2.5-pro
SMALL_MODEL=gemini-2.5-flash
```
</details>

<details>
<summary><b>Anthropic pass-through</b> — logging / middleware only</summary>

```env
PREFERRED_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```
</details>

---

## How It Works

### Request Flow

```
  ┌─────────────┐       ┌──────────────────────────────────────┐       ┌──────────────┐
  │ Claude Code │       │            Proxy Pipeline            │       │ Your Backend │
  │             │       │                                      │       │              │
  │  Anthropic  │──────►│  1. Validate & map model name        │──────►│   OpenAI     │
  │  Messages   │       │  2. Convert content blocks           │       │   Chat       │
  │  API        │◄──────│  3. Sanitize schemas                 │◄──────│   Completions│
  │             │       │  4. Stream + normalize tool calls    │       │   API        │
  └─────────────┘       │  5. Repair JSON, strip think tags    │       └──────────────┘
                        └──────────────────────────────────────┘
```

### Tool Call Normalization

The proxy intercepts the response stream and normalizes tool calls from **7 different formats** into standard Anthropic `tool_use` blocks:

| Dialect | Format | Detection |
|:--------|:-------|:----------|
| **Native** | OpenAI `delta.tool_calls` | Structured API field |
| **DeepSeek** | DSML XML tags | `<\|DSML\|invoke ...>` |
| **MiniMax** | Custom XML | `<minimax:tool_call>` |
| **GLM** | Bracket notation | `[ToolName]` or `[Tool: Name]` |
| **Hermes** | XML-wrapped JSON | `<tool_call>{...}</tool_call>` |
| **Qwen** | Semi-structured text | `Tool: ...\nArguments: ...` |
| **Fallback** | JSON in code blocks | `` ```json{...}``` `` |

### Streaming FSM

A finite state machine processes every SSE chunk in real time:

```
    ┌───────────────────┐          tag           ┌───────────────────┐
    │                   │       detected          │                   │
    │    PASSTHROUGH     │ ─────────────────────► │    BUFFERING      │
    │                   │                         │                   │
    │  • Emit safe text │                         │  • Accumulate     │
    │  • 35-byte overlap│ ◄───────────────────── │  • Try 7 parsers  │
    │  • Watch for tags │      parsed / failed    │  • JSON repair    │
    │                   │                         │  • Salvage logic  │
    └───────────────────┘                         └───────────────────┘
```

**Key features:**
- **Overlap buffer** — catches tool tags split across chunk boundaries
- **Dynamic detection** — watches for tool names from the current request
- **Brace-matching** — handles deeply nested JSON arguments correctly
- **Salvage mode** — extracts tool calls from malformed output as last resort
- **Think-tag stripping** — filters `<think>`/`</think>` reasoning tokens

---

## API

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/v1/messages` | Full Anthropic Messages API — streaming & non-streaming, tool use, system prompts, multi-turn |
| `POST` | `/v1/messages/count_tokens` | Token counting (approximate) |
| `GET`  | `/` | Health check — returns `{"status": "ok"}` |

---

## Wrapper Script

For isolated sessions that won't interfere with existing Claude Code configuration:

```bash
#!/usr/bin/env bash
# Save as ~/.local/bin/claudep (or anywhere on PATH)
exec env \
  ANTHROPIC_AUTH_TOKEN="placeholder" \
  ANTHROPIC_BASE_URL="http://localhost:8082" \
  API_TIMEOUT_MS="3000000" \
  claude "$@"
```

---

## Testing

```bash
# Proxy-only — 14 tests, no API key needed
PROXY_ONLY=1 PROXY_BASE_URL=http://localhost:8082 \
  uv run pytest tests/tests.py -v

# Full suite — 21 tests, compares proxy output against real Anthropic API
ANTHROPIC_API_KEY=sk-ant-... PROXY_BASE_URL=http://localhost:8082 \
  uv run pytest tests/tests.py -v
```

---

## Monitoring (Optional)

Lightweight observability stack (~200MB total, runs on constrained hardware):

```bash
cd monitoring && docker compose up -d
```

| Service | Port | Purpose |
|:--------|:-----|:--------|
| Prometheus | `9090` | Metrics collection, 7-day retention |
| cAdvisor | `8080` | Container resource metrics |
| Loki | `3100` | Log aggregation |
| Promtail | — | Ships container logs to Loki |

---

## Project Structure

```
.
├── server.py              # Proxy server — translation, FSM, normalization
├── pyproject.toml          # Dependencies (uv-managed)
├── Dockerfile              # Production container (amd64 + arm64)
├── .env.example            # Configuration template
├── tests/
│   └── tests.py            # Test suite (21 tests)
└── monitoring/             # Prometheus + Loki + cAdvisor stack
```

## Known Limitations

- Some models switch between native and text-embedded tool call formats mid-conversation. All known variants are handled, but rare formats may need new parser support.
- Token counting is approximate (uses LiteLLM's tokenizer, not the backend model's).
- The `thinking` parameter is accepted but thinking content from non-Anthropic models is stripped.

## Contributing

Issues and pull requests are welcome. If you encounter a model that emits tool calls in an unrecognized format, please open an issue with the raw output and we'll add parser support.

## License

MIT
