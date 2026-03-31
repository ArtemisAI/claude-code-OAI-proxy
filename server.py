from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
            "Unclosed client session",
            "Unclosed connector",
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert FastAPI 422 validation errors to Anthropic-format 400 errors."""
    errors = exc.errors()
    # Build a human-readable message from pydantic errors
    parts = []
    for err in errors:
        loc = ".".join(str(x) for x in err["loc"] if x != "body")
        parts.append(f"{loc}: {err['msg']}")
    message = "; ".join(parts) if parts else "Invalid request"
    return JSONResponse(
        status_code=400,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": message,
            },
        },
    )


# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

def _safe_get(obj, key, default=None):
    """Safely get an attribute from a Pydantic model or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")
OPUS_MODEL = os.environ.get("OPUS_MODEL", BIG_MODEL)  # Defaults to BIG_MODEL if not set

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

# Helper function to clean schema for stricter providers (OpenAI, Gemini, vLLM)
def clean_tool_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for strict models."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by various strict schemas
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        schema.pop("$comment", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time", "uuid", "uri"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_tool_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_tool_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        if PREFERRED_PROVIDER == "anthropic":
            # Don't remap to big/small models, just add the prefix
            new_model = f"anthropic/{clean_v}"
            mapped = True

        # Map Opus to OPUS_MODEL based on provider preference
        elif 'opus' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and OPUS_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{OPUS_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{OPUS_MODEL}"
                mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Opus to OPUS_MODEL based on provider preference
        if 'opus' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and OPUS_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{OPUS_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{OPUS_MODEL}"
                mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def parse_deepseek_dsml(content: str) -> tuple:
    """
    Detects and extracts DeepSeek-V3.2 DSML tool calls from raw text.
    Handles <｜DSML｜function_calls> with nested <｜DSML｜invoke> and <｜DSML｜parameter> tags.
    Also handles the simpler <｜tool calls begin｜> variant.
    Returns (cleaned_text, synthesized_tool_calls).
    """
    synthesized_tool_calls = []
    
    # --- Variant 1: Full DSML with invoke/parameter tags ---
    dsml_bot = "<｜DSML｜function_calls>"
    dsml_eot = "</｜DSML｜function_calls>"
    
    if dsml_bot in content:
        dsml_pattern = re.compile(
            rf'{re.escape(dsml_bot)}(.*?){re.escape(dsml_eot)}', re.DOTALL
        )
        invoke_pattern = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
        )
        parameter_pattern = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>', re.DOTALL
        )
        
        dsml_blocks = dsml_pattern.findall(content)
        content = dsml_pattern.sub('', content).strip()
        
        for block in dsml_blocks:
            invocations = invoke_pattern.findall(block)
            for func_name, param_block in invocations:
                parameters = {}
                params = parameter_pattern.findall(param_block)
                
                for param_name, is_string, param_value in params:
                    param_value = param_value.strip()
                    if is_string.lower() == "true":
                        parameters[param_name] = param_value
                    else:
                        try:
                            parameters[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            parameters[param_name] = param_value
                
                synthesized_tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": func_name.strip(),
                        "arguments": json.dumps(parameters, ensure_ascii=False)
                    }
                })
    
    # --- Variant 2: Simpler <｜tool calls begin｜> wrapper with JSON body ---
    simple_bot = "<｜tool calls begin｜>"
    simple_eot = "<｜tool calls end｜>"
    
    if simple_bot in content:
        simple_pattern = re.compile(
            rf'{re.escape(simple_bot)}.*?<｜tool call begin｜>(.*?)<｜tool call end｜>.*?{re.escape(simple_eot)}',
            re.DOTALL
        )
        for match in simple_pattern.finditer(content):
            tool_call_json = match.group(1).strip()
            try:
                parsed = json.loads(tool_call_json)
                synthesized_tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": parsed.get("name", ""),
                        "arguments": json.dumps(parsed.get("arguments", {}))
                    }
                })
            except (json.JSONDecodeError, Exception):
                pass
        content = simple_pattern.sub('', content).strip()
    
    return content, synthesized_tool_calls


def parse_minimax_xml(content: str) -> tuple:
    """
    Detects and extracts MiniMax-M2.5 tool calls from raw text.
    Handles <minimax:tool_call> with nested <invoke> and <parameter> tags.
    Applies heuristic type casting since MiniMax lacks explicit type attributes.
    Returns (cleaned_text, synthesized_tool_calls).
    """
    bot_token = "<minimax:tool_call>"
    eot_token = "</minimax:tool_call>"
    
    if bot_token not in content:
        return content, []
    
    synthesized_tool_calls = []
    
    block_pattern = re.compile(
        rf'{re.escape(bot_token)}(.*?){re.escape(eot_token)}', re.DOTALL
    )
    invoke_pattern = re.compile(
        r'<invoke\s+name="?([^">]+)"?>(.*?)</invoke>', re.DOTALL
    )
    param_pattern = re.compile(
        r'<parameter\s+name="?([^">]+)"?>(.*?)</parameter>', re.DOTALL
    )
    
    blocks = block_pattern.findall(content)
    cleaned_content = block_pattern.sub('', content).strip()
    
    for block in blocks:
        invocations = invoke_pattern.findall(block)
        for func_name, param_block in invocations:
            parameters = {}
            params = param_pattern.findall(param_block)
            
            for param_name, param_value in params:
                val = param_value.strip()
                # Heuristic type casting
                if val.lower() in ["true", "false"]:
                    parameters[param_name] = val.lower() == "true"
                elif val.isdigit():
                    parameters[param_name] = int(val)
                else:
                    try:
                        parameters[param_name] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        parameters[param_name] = val
            
            synthesized_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": func_name.strip(),
                    "arguments": json.dumps(parameters, ensure_ascii=False)
                }
            })
    
    return cleaned_content, synthesized_tool_calls


def parse_tool_call_xml(content: str) -> tuple:
    """
    Detects and extracts <tool_call> XML dialect tool calls.
    Handles two variants:
      1) JSON body: <tool_call>{"name": "Read", "arguments": {...}}</tool_call>
      2) XML body: <tool_call><tool_name>Read</tool_name><arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    Returns (cleaned_text, synthesized_tool_calls).
    """
    if "<tool_call>" not in content:
        return content, []

    synthesized_tool_calls = []
    block_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    blocks = block_pattern.findall(content)
    cleaned_content = block_pattern.sub('', content).strip()

    for block in blocks:
        block = block.strip()
        # Try JSON body first
        try:
            data = json.loads(block)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            synthesized_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                }
            })
            continue
        except (json.JSONDecodeError, ValueError):
            pass

        # Try XML body with <tool_name>, <arg_key>, <arg_value>
        name_match = re.search(r'<tool_name>(.*?)</tool_name>', block)
        if name_match:
            func_name = name_match.group(1).strip()
            keys = re.findall(r'<arg_key>(.*?)</arg_key>', block)
            values = re.findall(r'<arg_value>(.*?)</arg_value>', block, re.DOTALL)
            parameters = {}
            for k, v in zip(keys, values):
                v = v.strip()
                if v.lower() in ["true", "false"]:
                    parameters[k] = v.lower() == "true"
                elif v.isdigit():
                    parameters[k] = int(v)
                else:
                    try:
                        parameters[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        parameters[k] = v
            synthesized_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(parameters)
                }
            })

    return cleaned_content, synthesized_tool_calls


def parse_qwen_text(content: str) -> tuple:
    """
    Detects and extracts Qwen semi-structured plain-text tool calls.
    Handles patterns like:
        Tool usage:
        Tool: function_name
        Arguments: {"key": "value"}
    Also matches 'Input:' as an alias for 'Arguments:' (gateway passthrough variant).
    Returns (cleaned_text, synthesized_tool_calls).
    """
    if "Tool usage:" not in content and "Tool:" not in content:
        return content, []
    
    synthesized_tool_calls = []
    
    # Regex to capture tool name and JSON argument block
    # Matches both "Arguments:" and "Input:" to handle gateway passthrough format
    pattern = re.compile(r'Tool:\s*([^\n]+)\n+(?:Arguments|Input):\s*(\{.*\})', re.DOTALL)
    matches = pattern.findall(content)
    
    # Excise matched tool invocation strings from primary content
    cleaned_content = pattern.sub('', content).replace("Tool usage:", "").strip()
    
    for func_name, arguments in matches:
        try:
            json.loads(arguments)  # Validate JSON
            synthesized_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": func_name.strip(),
                    "arguments": arguments.strip()
                }
            })
        except json.JSONDecodeError:
            # Attempt repair before dropping
            try:
                import json_repair
                repaired = json_repair.repair_json(arguments, return_objects=True)
                if isinstance(repaired, dict):
                    synthesized_tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": func_name.strip(),
                            "arguments": json.dumps(repaired)
                        }
                    })
            except Exception:
                continue
    
    return cleaned_content, synthesized_tool_calls


def parse_bracket_tool_call(content: str) -> tuple:
    """
    Catches tool calls in the [ToolName] format emitted by some models (e.g., GLM):
        [Write]
        file_path: /tmp/hello.py
        content: def main():
            print("hello")
    Also handles JSON body variant:
        [Bash]
        {"command": "echo hello"}
    Returns (cleaned_text, synthesized_tool_calls).
    """
    # Match [ToolName] where ToolName is a single word (capitalized, no spaces)
    pattern = re.compile(
        r'\[([A-Z][a-zA-Z_]*)\]\s*\n(.*?)(?=\n\[[A-Z]|\Z)',
        re.DOTALL
    )
    matches = pattern.findall(content)
    if not matches:
        return content, []

    synthesized_tool_calls = []
    cleaned_content = content

    for func_name, body in matches:
        func_name = func_name.strip()
        body = body.strip()
        if not body:
            continue

        arguments = None
        # Try JSON body first
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                arguments = body
        except (json.JSONDecodeError, ValueError):
            pass

        # Try key: value format
        if arguments is None:
            params = {}
            current_key = None
            current_value_lines = []
            for line in body.split('\n'):
                # Check if line starts a new key (word followed by colon, not indented)
                key_match = re.match(r'^([a-z_][a-z_0-9]*)\s*:\s*(.*)', line)
                if key_match and not line.startswith(' ') and not line.startswith('\t'):
                    if current_key is not None:
                        params[current_key] = '\n'.join(current_value_lines).strip()
                    current_key = key_match.group(1)
                    current_value_lines = [key_match.group(2)]
                elif current_key is not None:
                    current_value_lines.append(line)
            if current_key is not None:
                params[current_key] = '\n'.join(current_value_lines).strip()

            if params:
                # Type-cast values
                for k, v in params.items():
                    if v.lower() in ('true', 'false'):
                        params[k] = v.lower() == 'true'
                    elif v.isdigit():
                        params[k] = int(v)
                    else:
                        try:
                            params[k] = json.loads(v)
                        except (json.JSONDecodeError, ValueError):
                            pass  # keep as string
                arguments = json.dumps(params)

        if arguments:
            # Remove the matched block from content
            block_pattern = re.compile(
                r'\[' + re.escape(func_name) + r'\]\s*\n' + re.escape(body),
                re.DOTALL
            )
            cleaned_content = block_pattern.sub('', cleaned_content, count=1)
            synthesized_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments
                }
            })

    return cleaned_content.strip(), synthesized_tool_calls


def parse_gateway_passthrough(content: str) -> tuple:
    """
    Catches tool calls emitted in the NewAPI/LiteLLM gateway passthrough format.
    Handles three variants:
      1) [Tool: name]\nInput: {"arg": "value"}
      2) [Tool: name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>  (hybrid XML)
      3) [Tool: name]\nArguments: {"arg": "value"}
    Returns (cleaned_text, synthesized_tool_calls).
    """
    if "[Tool:" not in content and "[Tool ]" not in content:
        return content, []

    synthesized_tool_calls = []
    cleaned_content = content

    # Variant 2: Hybrid [Tool: Name<arg_key>...<arg_value>...</tool_call>
    hybrid_pattern = re.compile(
        r'\[Tool:\s*(\w+)((?:<arg_key>.*?</arg_value>)+)\s*(?:</tool_call>)?',
        re.DOTALL
    )
    hybrid_matches = hybrid_pattern.findall(cleaned_content)
    if hybrid_matches:
        cleaned_content = hybrid_pattern.sub('', cleaned_content).strip()
        for func_name, xml_body in hybrid_matches:
            func_name = func_name.strip()
            keys = re.findall(r'<arg_key>(.*?)</arg_key>', xml_body)
            values = re.findall(r'<arg_value>(.*?)</arg_value>', xml_body, re.DOTALL)
            params = {}
            for k, v in zip(keys, values):
                v = v.strip()
                if v.lower() in ('true', 'false'):
                    params[k] = v.lower() == 'true'
                elif v.isdigit():
                    params[k] = int(v)
                else:
                    try:
                        params[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        params[k] = v
            synthesized_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(params)
                }
            })

    # Variant 1 & 3: [Tool: name]\nInput/Arguments: {json}
    # Use brace-matching instead of regex for the JSON body (handles nested braces in content)
    header_pattern = re.compile(r'\[Tool:?\s*([^\]<]+)\]\s*\n+(?:Input|Arguments):\s*')
    remaining = cleaned_content
    new_cleaned = ""
    while True:
        m = header_pattern.search(remaining)
        if not m:
            new_cleaned += remaining
            break
        func_name = m.group(1).strip()
        new_cleaned += remaining[:m.start()]
        json_start = m.end()
        # Brace-match to find the complete JSON object
        if json_start < len(remaining) and remaining[json_start] == '{':
            depth = 0
            in_string = False
            escape = False
            json_end = json_start
            for i in range(json_start, len(remaining)):
                c = remaining[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_end = i + 1
                            break
            if depth == 0:
                arguments = remaining[json_start:json_end]
                try:
                    json.loads(arguments)
                    synthesized_tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": arguments
                        }
                    })
                    remaining = remaining[json_end:]
                    continue
                except json.JSONDecodeError:
                    try:
                        import json_repair
                        repaired = json_repair.repair_json(arguments, return_objects=True)
                        if isinstance(repaired, dict):
                            synthesized_tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:16]}",
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": json.dumps(repaired)
                                }
                            })
                            remaining = remaining[json_end:]
                            continue
                    except Exception:
                        pass
        # If we couldn't parse, skip this match and continue
        new_cleaned += remaining[m.start():m.end()]
        remaining = remaining[m.end():]
    cleaned_content = new_cleaned.strip()

    return cleaned_content, synthesized_tool_calls


def parse_generic_json_blocks(content: str) -> tuple:
    """
    Universal fallback: detects ```json code blocks containing tool-call-like structures.
    Returns (cleaned_text, synthesized_tool_calls).
    """
    json_block_pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
    synthesized_tool_calls = []
    blocks_to_remove = []
    
    for match in json_block_pattern.finditer(content):
        json_str = match.group(1).strip()
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                name = (parsed.get("name") or parsed.get("action") or 
                        parsed.get("tool") or parsed.get("function"))
                args = (parsed.get("arguments") or parsed.get("args") or 
                        parsed.get("parameters") or parsed.get("action_input"))
                
                if name and args is not None:
                    synthesized_tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args)
                        }
                    })
                    blocks_to_remove.append(match.group(0))
        except (json.JSONDecodeError, Exception):
            pass
    
    for block in blocks_to_remove:
        content = content.replace(block, "")
    
    return content.strip(), synthesized_tool_calls


def normalize_tool_calls(content_text: str, existing_tool_calls: list) -> tuple:
    """
    Unified lexical normalization pipeline (doc §4.4).
    Sequentially applies DeepSeek, MiniMax, Qwen, and generic JSON parsers,
    aggregating discovered tool calls with any natively parsed ones.
    """
    if not content_text:
        return "", existing_tool_calls if existing_tool_calls else []
    
    tools = list(existing_tool_calls) if existing_tool_calls else []
    
    content_text, dsml_tools = parse_deepseek_dsml(content_text)
    tools.extend(dsml_tools)
    
    content_text, minimax_tools = parse_minimax_xml(content_text)
    tools.extend(minimax_tools)

    content_text, tool_call_xml_tools = parse_tool_call_xml(content_text)
    tools.extend(tool_call_xml_tools)

    # Parse [ToolName]\nkey: value format (GLM, some fine-tuned models)
    content_text, bracket_tools = parse_bracket_tool_call(content_text)
    tools.extend(bracket_tools)

    # Run gateway passthrough parser (more specific than Qwen due to brackets)
    content_text, gateway_tools = parse_gateway_passthrough(content_text)
    tools.extend(gateway_tools)

    content_text, qwen_tools = parse_qwen_text(content_text)
    tools.extend(qwen_tools)

    # Only run generic fallback if no dialect-specific tools were found
    if not dsml_tools and not minimax_tools and not qwen_tools and not gateway_tools and not bracket_tools:
        content_text, generic_tools = parse_generic_json_blocks(content_text)
        tools.extend(generic_tools)
    
    return content_text.strip(), tools

class TextStreamFSM:
    def __init__(self, tool_names=None):
        self.state = 0 # 0=TEXT, 1=BUFFERING
        self.buffer = ""
        self.overlap_buffer = ""
        self.overlap_size = 35 # safe margin
        self.start_tags = [
            "<｜DSML｜invoke",
            "<minimax:tool_call>",
            "<｜tool calls begin｜>",
            "<tool_call>",
            "<think>",
            "[Tool:",
            "[Function:",
            "```json"
        ]
        # Dynamically add [ToolName] patterns for tools the model knows about
        if tool_names:
            for name in tool_names:
                tag = f"[{name}]"
                if tag not in self.start_tags:
                    self.start_tags.append(tag)
        
    def process_chunk(self, text_chunk: str) -> tuple:
        """Returns (text_to_yield, list_of_parsed_tools)"""
        text_to_yield = ""
        tools_to_yield = []
        
        combined = self.overlap_buffer + text_chunk
        
        if self.state == 0:
            first_idx = len(combined)
            matched_tag = None
            for tag in self.start_tags:
                idx = combined.find(tag)
                if idx != -1 and idx < first_idx:
                    first_idx = idx
                    matched_tag = tag
            
            if matched_tag is not None:
                pre_text = combined[:first_idx]
                if pre_text:
                    text_to_yield += pre_text
                    
                self.state = 1
                self.buffer = combined[first_idx:]
                self.overlap_buffer = ""
            else:
                if len(combined) > self.overlap_size:
                    safe_text = combined[:-self.overlap_size]
                    self.overlap_buffer = combined[-self.overlap_size:]
                    if safe_text:
                        text_to_yield += safe_text
                else:
                    self.overlap_buffer = combined
                    
        elif self.state == 1:
            self.buffer += text_chunk

            # Handle <think>...</think> blocks — strip entirely
            if self.buffer.lstrip().startswith('<think>'):
                if '</think>' in self.buffer:
                    # Complete think block — discard it, resume normal processing
                    after_think = self.buffer.split('</think>', 1)[1]
                    self.state = 0
                    self.buffer = ""
                    self.overlap_buffer = ""
                    if after_think.strip():
                        sub_text, sub_tools = self.process_chunk(after_think)
                        text_to_yield += sub_text
                        tools_to_yield.extend(sub_tools)
                    return text_to_yield, tools_to_yield
                else:
                    # Incomplete think block — keep buffering
                    return text_to_yield, tools_to_yield

            clean_text, parsed_tools = normalize_tool_calls(self.buffer, [])

            if parsed_tools:
                tools_to_yield.extend(parsed_tools)
                self.state = 0
                self.buffer = ""
                # Feed leftover text back through to catch consecutive tools or resume standard text
                if clean_text:
                    sub_text, sub_tools = self.process_chunk(clean_text)
                    text_to_yield += sub_text
                    tools_to_yield.extend(sub_tools)
                    
        return text_to_yield, tools_to_yield
        
    def _salvage_tool_call(self, buf: str) -> list:
        """Last-resort extraction from malformed hybrid tool call content.
        Handles cases like: [Tool: Bash]\nInput: {"command": "cmd...</arg_value>...</tool_call>
        """
        import json_repair as jr
        tools = []

        # Try to extract function name from [Tool: Name] or [Name]
        name_match = re.match(r'\[(?:Tool:\s*)?(\w+)\]', buf)
        if not name_match:
            return []
        func_name = name_match.group(1)

        # Strategy 1: Try to find and repair truncated JSON after Input:/Arguments:
        json_match = re.search(r'(?:Input|Arguments):\s*(\{.*)', buf, re.DOTALL)
        if json_match:
            raw_json = json_match.group(1)
            # Strip trailing XML tags
            raw_json = re.sub(r'</?(arg_key|arg_value|tool_call)>.*', '', raw_json, flags=re.DOTALL).strip()
            # Try to repair truncated JSON
            try:
                repaired = jr.repair_json(raw_json, return_objects=True)
                if isinstance(repaired, dict) and repaired:
                    tools.append({
                        "id": f"call_{uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {"name": func_name, "arguments": json.dumps(repaired)}
                    })
                    return tools
            except Exception:
                pass

        # Strategy 2: Extract from <arg_key>/<arg_value> pairs
        keys = re.findall(r'<arg_key>(.*?)</arg_key>', buf)
        values = re.findall(r'<arg_value>(.*?)</arg_value>', buf, re.DOTALL)
        if keys and values:
            params = {k: v.strip() for k, v in zip(keys, values)}
            tools.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {"name": func_name, "arguments": json.dumps(params)}
            })
            return tools

        # Strategy 3: If there's a truncated JSON with a recognizable key, try to complete it
        if json_match:
            raw = json_match.group(1).strip()
            # Extract the first key-value pair at minimum
            kv_match = re.search(r'"(\w+)":\s*"([^"]*)"', raw)
            if kv_match:
                tools.append({
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": func_name, "arguments": json.dumps({kv_match.group(1): kv_match.group(2)})}
                })
                return tools

        return []

    def flush(self) -> tuple:
        text_to_yield = ""
        tools_to_yield = []
        if self.state == 1 and self.buffer:
            clean_text, parsed_tools = normalize_tool_calls(self.buffer, [])
            if parsed_tools:
                tools_to_yield.extend(parsed_tools)
                text_to_yield += clean_text
            else:
                # If EOF occurs during buffering and tool parsing fails, check if it looks like tool content
                buf = self.buffer.strip()
                is_tool_content = (
                    (buf.startswith('<') and ('tool_call' in buf or 'invoke' in buf or 'DSML' in buf or 'arg_key' in buf)) or
                    (buf.startswith('[Tool:') or buf.startswith('[Function:')) or
                    any(buf.startswith(f'[{tag}]') for tag in ['Bash', 'Write', 'Read', 'Edit', 'Glob', 'Grep', 'WebSearch', 'WebFetch'])
                )
                if is_tool_content:
                    # Last-resort: try to extract tool call from malformed hybrid content
                    salvaged = self._salvage_tool_call(buf)
                    if salvaged:
                        tools_to_yield.extend(salvaged)
                    else:
                        logger.warning(f"FSM flush: dropping unrecognized tool call content ({len(self.buffer)} chars): {self.buffer[:200]!r}")
                else:
                    # Not tool content — safe to emit as text
                    text_to_yield += self.buffer
                
        elif self.state == 0 and self.overlap_buffer:
            text_to_yield += self.overlap_buffer
            
        return text_to_yield, tools_to_yield


def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(_safe_get(block, "type") == "tool_result" for block in content):
                # For user messages with tool_result, split into separate messages
                accumulated_text_parts = []
                
                for block in content:
                    block_type = _safe_get(block, "type")
                     
                    if block_type == "text":
                        text_content = _safe_get(block, "text", "")
                        accumulated_text_parts.append(text_content)
                         
                    elif block_type == "tool_result":
                        # Add tool result as an explicit message
                        tool_id = _safe_get(block, "tool_use_id", "")
                        is_error = _safe_get(block, "is_error", False)
                         
                        # Handle different formats of tool result content
                        result_content = ""
                        content_payload = _safe_get(block, "content")
                         
                        if isinstance(content_payload, str):
                            result_content = content_payload
                        elif isinstance(content_payload, list):
                            for item in content_payload:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    result_content += item.get("text", "") + "\n"
                                elif _safe_get(item, "type") == "text":
                                    result_content += _safe_get(item, "text", "") + "\n"
                                else:
                                    result_content += str(item) + "\n"
                        elif isinstance(content_payload, dict):
                            if content_payload.get("type") == "text":
                                result_content = content_payload.get("text", "")
                            else:
                                try:
                                    result_content = json.dumps(content_payload)
                                except:
                                    result_content = str(content_payload)
                        else:
                            try:
                                result_content = str(content_payload)
                            except:
                                result_content = "Unparseable content"
                                
                        # To maintain conversational sequence, any text accumulated BEFORE 
                        # this tool result must be appended as a discrete user message.
                        if accumulated_text_parts:
                            messages.append({
                                "role": "user", 
                                "content": "\n".join(accumulated_text_parts).strip()
                            })
                            accumulated_text_parts = []
                            
                        # Synthesize the strictly compliant OpenAI Tool Role Message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result_content.strip()
                        })
                
                # If text parts remain after processing all tool results
                if accumulated_text_parts:
                    messages.append({
                        "role": "user",
                        "content": "\n".join(accumulated_text_parts).strip()
                    })
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Only include thinking field for Anthropic models
    if anthropic_request.thinking and anthropic_request.model.startswith("anthropic/"):
        litellm_request["thinking"] = anthropic_request.thinking

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_strict_model = not anthropic_request.model.startswith("anthropic/")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a strict non-Anthropic model
            input_schema = tool_dict.get("input_schema", {})
            if is_strict_model:
                 logger.debug(f"Cleaning schema for tool: {tool_dict.get('name')}")
                 input_schema = clean_tool_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Apply dialect normalizers to extract embedded JSON objects manually when strictly needed
        content_text, tool_calls = normalize_tool_calls(content_text, tool_calls)
        if tool_calls and finish_reason != "tool_calls":
            finish_reason = "tool_calls"
        
        # Create content list for Anthropic format
        content = []
        
        # Add tool calls if present (tool_use in Anthropic format)
        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments[:100]}... Attempting repair.")
                        try:
                            import json_repair
                            arguments = json_repair.repair_json(arguments, return_objects=True)
                        except Exception as repair_e:
                            logger.error(f"JSON repair failed: {str(repair_e)}")
                            arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif content_text and hasattr(original_request, 'tools') and original_request.tools:
            # Parser fallback for manual tool calls in content_text (ISSUE-004)
            # This handles models that return:
            #   preamble...
            #   tool_name
            #   {"arg": "value"}
            for tool in original_request.tools:
                tool_name = (tool.get("name") if isinstance(tool, dict) 
                           else getattr(tool, "name", ""))
                if not tool_name: continue
                
                # Look for tool_name followed by { ... }
                pattern = rf"(?:^|\n)\s*{re.escape(tool_name)}\s*\n\s*(\{{.*\}})"
                match = re.search(pattern, content_text, re.DOTALL)
                if match:
                    potential_json = match.group(1).strip()
                    args = None
                    # Try to find the valid JSON object by iteratively checking prefixes
                    for i in range(len(potential_json), 0, -1):
                        try:
                            if not potential_json[:i].strip().endswith("}"):
                                continue
                            args = json.loads(potential_json[:i])
                            break
                        except json.JSONDecodeError:
                            continue
                    
                    if args is not None:
                        logger.info(f"✨ Found manual tool call in text: {tool_name}")
                        # Update content_text to keep only the preamble
                        preamble = content_text[:match.start()].strip()
                        content_text = preamble
                        
                        content.append({
                            "type": "tool_use",
                            "id": f"tool_{uuid.uuid4()}",
                            "name": tool_name,
                            "input": args
                        })
                        # Overwrite finish_reason to fix stop_reason mapping later
                        finish_reason = "tool_calls"
                        break

        # Add text content block if present (text might be None or empty for pure tool call responses)
        # We add this AFTER tool processing to use the potentially cleaned/shortened content_text
        if content_text is not None and content_text != "":
            # Insert at the beginning of content list so it appears before tool_use
            content.insert(0, {"type": "text", "text": content_text})
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest, raw_request: Request = None):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': getattr(original_request, 'original_model', original_request.model) or original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # JSON Accumulation Buffer for tool arguments
        json_buffer = ""
        import time
        # Extract tool names for dynamic FSM start_tags
        tool_names = []
        if hasattr(original_request, 'tools') and original_request.tools:
            for t in original_request.tools:
                name = t.name if hasattr(t, 'name') else t.get('name', '') if isinstance(t, dict) else ''
                if name:
                    tool_names.append(name)
        text_fsm = TextStreamFSM(tool_names=tool_names)
        last_yield_time = time.time()
        next_embedded_tool_index = 1000

        # Process each chunk
        try:
          async for chunk in response_generator:
            try:
                # Check for client disconnection
                if raw_request and await raw_request.is_disconnected():
                    logger.warning("Client disconnected from streaming request.")
                    # We can break here to stop upstream generation
                    break

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Heartbeat Keepalive
                    if time.time() - last_yield_time > 10:
                        yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"
                        last_yield_time = time.time()

                    # Process native tool calls early to intercept
                    delta_tool_calls = None
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                        
                    # FSM Text Interception
                    if delta_content is not None and delta_content != "":
                        safe_text, embedded_tools = text_fsm.process_chunk(delta_content)
                        accumulated_text += delta_content # Keep for EOF fallback parsing
                        
                        # Inject embedded tools into streaming pipeline
                        if embedded_tools:
                            if delta_tool_calls is None:
                                delta_tool_calls = []
                            elif not isinstance(delta_tool_calls, list):
                                delta_tool_calls = list(delta_tool_calls)
                                
                            for t in embedded_tools:
                                delta_tool_calls.append({
                                    'index': next_embedded_tool_index,
                                    'id': t['id'],
                                    'function': {
                                        'name': t['function']['name'],
                                        'arguments': t['function']['arguments']
                                    }
                                })
                                next_embedded_tool_index += 1
                                
                        # Strip any residual think tags from safe text
                        if safe_text:
                            safe_text = re.sub(r'</?think>', '', safe_text)

                        # Stateful Text Emission from FSM safe text
                        if safe_text and tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': safe_text}})}\n\n"
                            last_yield_time = time.time()
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # Drain FSM overlap buffer before closing text block
                            if not text_block_closed:
                                pending_text, _ = text_fsm.flush()
                                if pending_text:
                                    text_sent = True
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': pending_text}})}\n\n"
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, flush the FSM (not raw accumulated_text
                            # which may contain tool call XML fragments)
                            elif not text_sent and not text_block_closed:
                                pending_text, _ = text_fsm.flush()
                                if pending_text:
                                    text_sent = True
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': pending_text}})}\n\n"
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - reset json_buffer and create a new tool_use block
                                json_buffer = ""
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Add to JSON Accumulation Buffer
                                if isinstance(arguments, str):
                                    json_buffer += arguments
                                else:
                                    json_buffer = json.dumps(arguments)

                                # Try to detect if buffered arguments are valid JSON
                                try:
                                    json.loads(json_buffer)
                                    # If it succeeds, it's valid JSON. Yield the accumulated buffer and reset it.
                                    args_json = json_buffer
                                    
                                    # Add to accumulated tool content
                                    tool_content += args_json
                                    
                                    # Send the update
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                                    
                                    # Reset buffer after yielding
                                    json_buffer = ""
                                except (json.JSONDecodeError, TypeError):
                                    # If it's incomplete JSON, do not yield. Wait for the next chunk to complete it.
                                    pass
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Flush the FSM
                        flushed_text, flushed_tools = text_fsm.flush()
                        if flushed_text and tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': flushed_text}})}\n\n"
                        
                        # === FIX 2+3: Final sweep for text-embedded tool calls ===
                        # When stream ends, models may have emitted tool calls as text.
                        late_tools = flushed_tools
                        cleaned_text = accumulated_text
                        
                        if accumulated_text and tool_index is None and not late_tools:
                            # Also try the fallback tool-name parser (ISSUE-004 pattern)
                            if not late_tools and hasattr(original_request, 'tools') and original_request.tools:
                                for tool in original_request.tools:
                                    tool_name = (tool.get("name") if isinstance(tool, dict) 
                                               else getattr(tool, "name", ""))
                                    if not tool_name: continue
                                    
                                    pattern = rf"(?:^|\n)\s*{re.escape(tool_name)}\s*\n\s*(\{{.*\}})"
                                    match = re.search(pattern, accumulated_text, re.DOTALL)
                                    if match:
                                        potential_json = match.group(1).strip()
                                        args = None
                                        for i in range(len(potential_json), 0, -1):
                                            try:
                                                if not potential_json[:i].strip().endswith("}"):
                                                    continue
                                                args = json.loads(potential_json[:i])
                                                break
                                            except json.JSONDecodeError:
                                                continue
                                        
                                        if args is not None:
                                            logger.info(f"✨ Streaming: Found late tool call in text: {tool_name}")
                                            cleaned_text = accumulated_text[:match.start()].strip()
                                            late_tools.append({
                                                "id": f"call_{uuid.uuid4().hex[:16]}",
                                                "type": "function",
                                                "function": {
                                                    "name": tool_name,
                                                    "arguments": json.dumps(args)
                                                }
                                            })
                                            break
                        
                        if late_tools:
                            # We found embedded tool calls — rewrite the stream ending
                            logger.info(f"🔧 Streaming: Intercepted {len(late_tools)} text-embedded tool call(s)")
                            
                            # Emit cleaned text (if any) before tool calls
                            if not text_block_closed:
                                if cleaned_text and cleaned_text.strip():
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': cleaned_text}})}\n\n"
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                text_block_closed = True
                            
                            # Emit each discovered tool as proper Anthropic tool_use blocks
                            for t in late_tools:
                                last_tool_index += 1
                                t_index = last_tool_index
                                func = t.get("function", {})
                                t_name = func.get("name", "")
                                t_id = t.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                                t_args = func.get("arguments", "{}")
                                
                                # content_block_start
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': t_index, 'content_block': {'type': 'tool_use', 'id': t_id, 'name': t_name, 'input': {}}})}\n\n"
                                # content_block_delta with full JSON
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': t_index, 'delta': {'type': 'input_json_delta', 'partial_json': t_args}})}\n\n"
                                # content_block_stop
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': t_index})}\n\n"
                            
                            # Force stop_reason to tool_use
                            finish_reason = "tool_calls"
                        else:
                            # No late tools found — normal close path
                            # Flush any remaining json_buffer as final tool call arguments
                            if json_buffer and tool_index is not None:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': last_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': json_buffer}})}\n\n"
                                json_buffer = ""
                            # Close any open tool call blocks
                            if tool_index is not None:
                                for i in range(1, last_tool_index + 1):
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                            
                            # If we haven't closed the text block, flush FSM and close it
                            # (use FSM flush, not raw accumulated_text which may contain tool call fragments)
                            if not text_block_closed:
                                if not text_sent:
                                    pending_text, _ = text_fsm.flush()
                                    if pending_text:
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': pending_text}})}\n\n"
                                # Close the text block
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        finally:
          # Ensure upstream generator is closed to avoid "Unclosed client session" errors
          if hasattr(response_generator, 'aclose'):
              await response_generator.aclose()

        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # Validate required fields before forwarding upstream
        if not request.messages:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "messages: field required",
                    },
                },
            )
        if not request.max_tokens:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "max_tokens: field required",
                    },
                },
            )

        # print the body here
        body = await raw_request.body()

        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            # Use custom OpenAI base URL if configured
            if OPENAI_BASE_URL:
                litellm_request["api_base"] = OPENAI_BASE_URL
                logger.debug(f"Using OpenAI API key and custom base URL {OPENAI_BASE_URL} for model: {request.model}")
            else:
                logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            if USE_VERTEX_AUTH:
                litellm_request["vertex_project"] = VERTEX_PROJECT
                litellm_request["vertex_location"] = VERTEX_LOCATION
                litellm_request["custom_llm_provider"] = "vertex_ai"
                logger.debug(f"Using Gemini ADC with project={VERTEX_PROJECT}, location={VERTEX_LOCATION} and model: {request.model}")
            else:
                litellm_request["api_key"] = GEMINI_API_KEY
                logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")
        
        # For OpenAI models - sanitize request format
        # IMPORTANT: role="tool" messages are native OpenAI format and must NOT be flattened.
        # Only content arrays with complex Anthropic-style blocks need conversion.
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            for i, msg in enumerate(litellm_request["messages"]):
                # Skip role="tool" messages entirely - they are already in correct OpenAI format
                if msg.get("role") == "tool":
                    # Ensure content is a string (OpenAI requires it)
                    if isinstance(msg.get("content"), list):
                        text_parts = []
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                text_parts.append(block)
                            else:
                                text_parts.append(json.dumps(block) if isinstance(block, dict) else str(block))
                        litellm_request["messages"][i]["content"] = "\n".join(text_parts) or "..."
                    elif msg.get("content") is None:
                        litellm_request["messages"][i]["content"] = "..."
                    continue
                
                # Skip assistant messages with tool_calls - they are already correctly formed
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Ensure content is a string or None (not a list)
                    if isinstance(msg.get("content"), list):
                        text_parts = [b.get("text", "") for b in msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
                        litellm_request["messages"][i]["content"] = "\n".join(text_parts) if text_parts else None
                    continue
                
                # For user/system messages: flatten content arrays to strings
                if "content" in msg and isinstance(msg["content"], list):
                    text_parts = []
                    for block in msg["content"]:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "image":
                                text_parts.append("[Image content]")
                            elif block.get("type") == "tool_use":
                                # This shouldn't appear in user messages, but handle gracefully
                                tool_name = block.get("name", "unknown")
                                tool_input = json.dumps(block.get("input", {}))
                                text_parts.append(f"[Tool: {tool_name}]\nInput: {tool_input}")
                            else:
                                text_parts.append(json.dumps(block))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    
                    litellm_request["messages"][i]["content"] = "\n".join(text_parts).strip() or "..."
                elif msg.get("content") is None:
                    litellm_request["messages"][i]["content"] = "..."
                
                # Remove unsupported fields
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        del msg[key]
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request, raw_request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Helper function to safely serialize objects for JSON
        def sanitize_for_json(obj):
            """递归地清理对象使其可以JSON序列化"""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Log all error details with safe serialization
        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Prepare token counter arguments
            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }
            
            # Count tokens
            token_count = token_counter(**token_counter_args)
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
