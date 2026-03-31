FROM python:3.11-slim

WORKDIR /claude-code-proxy

# Install uv
RUN pip install --upgrade uv

# Copy dependency specifications first (cache-friendly layer)
COPY pyproject.toml uv.lock ./

# Install dependencies into the system Python (no virtualenv needed in container)
RUN uv pip install --system -r pyproject.toml

# Copy project code
COPY . .

# Start the proxy
EXPOSE 8082
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
