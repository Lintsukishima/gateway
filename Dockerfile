FROM python:3.11-slim

WORKDIR /app

# system deps（可选，先不装也行）
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml /app/ 2>/dev/null || true
COPY requirements.txt /app/ 2>/dev/null || true

# 兼容你用 requirements.txt 的情况
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY . /app

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
