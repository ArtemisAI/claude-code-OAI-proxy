# Claude Code OAI Proxy

Use [Claude Code](https://docs.anthropic.com/en/docs/claude-code) with **any model** — DeepSeek, Qwen, GLM, Kimi, MiniMax, Gemini, GPT, or anything behind an OpenAI-compatible API.

```
Claude Code ──► This Proxy ──► Any OpenAI-compatible API ──► Your models
 Anthropic       translates     your endpoint                  DeepSeek,
 Messages API    both ways      (vLLM, Ollama, OneAPI, etc.)   Qwen, GLM, …
```

## Why

Claude Code speaks only the Anthropic Messages API. If your models are served behind an OpenAI-compatible endpoint, they can't talk to each other.

This proxy sits in the middle and translates both directions in real time — requests from Anthropic format to OpenAI Chat Completions, and streamed responses back.

The hard part isn't the format conversion. It's **tool calls**. Different models emit tool calls in wildly different formats — DeepSeek uses custom XML, MiniMax uses its own XML, Qwen uses semi-structured text, GLM uses bracket notation, and some models switch formats mid-conversation. This proxy normalizes all of them into proper Anthropic `tool_use` blocks so Claude Code can reliably execute tools.

---

## Quick Start

### 1. Configure

```bash
git clone https://github.com/ArtemisAI/Claude-Code-OAI-Proxy.git
cd Claude-Code-OAI-Proxy
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.com/v1
BIG_MODEL=your-large-model
SMALL_MODEL=your-small-model
```

### 2. Run

```bash
# With uv (development)
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

# With Docker
docker build -t claude-oai-proxy .
docker run -d --env-file .env -p 8082:8082 --restart unless-stopped claude-oai-proxy

# Pre-built image (amd64 + arm64)
docker run -d --env-file .env -p 8082:8082 --restart unless-stopped \
  ghcr.io/artemisai/claude-code-oai-proxy:main
```

### 3. Connect Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

> **Note:** `ANTHROPIC_BASE_URL` must **not** include `/v1` — Claude Code appends it automatically.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for your backend |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Your API endpoint |
| `BIG_MODEL` | `gpt-4.1` | Model for claude-sonnet/opus requests |
| `SMALL_MODEL` | `gpt-4.1-mini` | Model for claude-haiku requests |
| `OPUS_MODEL` | same as `BIG_MODEL` | Model for claude-opus requests specifically |
| `PREFERRED_PROVIDER` | `openai` | Provider routing: `openai`, `google`, or `anthropic` |

### Model Mapping

The proxy remaps Claude model names to your configured models:

```
claude-haiku-*    →  SMALL_MODEL
claude-sonnet-*   →  BIG_MODEL
claude-opus-*     →  OPUS_MODEL
```

Direct model names (not matching `claude-*`) are passed through unchanged.

### Provider Examples

<details>
<summary><strong>OpenAI-compatible API</strong> (vLLM, Ollama, OneAPI, LiteLLM, etc.)</summary>

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-endpoint.com/v1
BIG_MODEL=your-large-model
SMALL_MODEL=your-small-model
```
</details>

<details>
<summary><strong>Google Gemini</strong> (AI Studio)</summary>

```env
PREFERRED_PROVIDER=google
GEMINI_API_KEY=your-key
BIG_MODEL=gemini-2.5-pro
SMALL_MODEL=gemini-2.5-flash
```
</details>

<details>
<summary><strong>Google Vertex AI</strong> (Application Default Credentials)</summary>

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
<summary><strong>Anthropic pass-through</strong> (proxy for logging/middleware)</summary>

```env
PREFERRED_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```
</details>

---

## How It Works

### Request Flow

```
1. /v1/messages request arrives (Anthropic format)
2. Validate request, map model name
3. Convert Anthropic content blocks → OpenAI messages
   - tool_use / tool_result → function calls
   - Sanitize JSON schemas for strict providers
4. Forward to your backend via LiteLLM
5. Stream response back: OpenAI deltas → Anthropic SSE
   - FSM intercepts and normalizes tool calls
   - JSON repair for malformed arguments
6. Claude Code receives clean Anthropic-format stream
```

### Tool Call Normalization

Different models express tool calls in incompatible formats. The proxy intercepts the text stream and normalizes them all:

| Model Family | Format | Example |
|-------------|--------|---------|
| **Native** | OpenAI `delta.tool_calls` | Standard structured response |
| **DeepSeek** | DSML XML | `<\|DSML\|invoke name="Bash">{"command":"ls"}</invoke>` |
| **MiniMax** | Custom XML | `<minimax:tool_call><invoke name="Bash">...</invoke>` |
| **GLM** | Bracket notation | `[Bash]\ncommand: ls` |
| **Hermes** | Tool call XML | `<tool_call>{"name":"Bash",...}</tool_call>` |
| **Qwen** | Semi-structured text | `Tool: Bash\nArguments: {"command":"ls"}` |
| **Fallback** | JSON code blocks | `` ```json{"name":"Bash",...}``` `` |

### Streaming FSM

A finite state machine monitors every text chunk in real time:

- **Overlap buffer** catches tool-call tags split across SSE chunks
- **Dynamic tag detection** based on the tools in each request
- **Brace-matching** correctly handles large nested JSON arguments
- **Salvage logic** extracts tool calls from malformed output as a last resort
- **Think-tag stripping** filters `<think>`/`</think>` reasoning tokens

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/messages` | Anthropic Messages API (streaming and non-streaming) |
| `POST` | `/v1/messages/count_tokens` | Token counting |
| `GET`  | `/`  | Health check |

---

## Wrapper Script

For isolated sessions that don't interfere with existing Claude Code configuration:

```bash
#!/usr/bin/env bash
# Save as ~/.local/bin/claudep (or anywhere on your PATH)
exec env \
  ANTHROPIC_AUTH_TOKEN="placeholder" \
  ANTHROPIC_BASE_URL="http://localhost:8082" \
  API_TIMEOUT_MS="3000000" \
  claude "$@"
```

---

## Testing

```bash
# Start the proxy, then:

# Proxy-only (14 tests, no external API key needed)
PROXY_ONLY=1 PROXY_BASE_URL=http://localhost:8082 \
  uv run pytest tests/tests.py -v

# Full suite (21 tests, validates against real Anthropic API)
ANTHROPIC_API_KEY=sk-ant-... PROXY_BASE_URL=http://localhost:8082 \
  uv run pytest tests/tests.py -v
```

---

## Monitoring (Optional)

Lightweight observability stack (~200MB, suitable for constrained environments):

```bash
cd monitoring && docker compose up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Metrics (7-day retention) |
| cAdvisor | 8080 | Container resource metrics |
| Loki | 3100 | Log aggregation |
| Promtail | — | Ships container logs to Loki |

---

## Project Structure

```
server.py              Proxy server
pyproject.toml         Dependencies (uv-managed)
Dockerfile             Production container (amd64 + arm64)
.env.example           Configuration template
tests/tests.py         Test suite (21 tests)
monitoring/            Prometheus + Loki + cAdvisor stack
```

## Known Limitations

- Some models switch between native and text-embedded tool call formats mid-conversation. All known variants are handled, but rare formats may occasionally need new parser support.
- Token counting is approximate (uses LiteLLM's tokenizer, not the backend model's native tokenizer).
- The `thinking` parameter is accepted but thinking content from non-Anthropic models is stripped.

## License

MIT
