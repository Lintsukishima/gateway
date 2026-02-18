#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] env file not found: $ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

base="${UPSTREAM_BASE_URL:-}"
key="${UPSTREAM_API_KEY:-}"
model="${UPSTREAM_MODEL:-openai/gpt-4o-mini}"

if [[ -z "$base" && -z "$key" ]]; then
  echo "[ERROR] UPSTREAM_BASE_URL / UPSTREAM_API_KEY 都为空，未成对配置"
  exit 1
fi
if [[ -z "$base" || -z "$key" ]]; then
  echo "[ERROR] UPSTREAM_BASE_URL / UPSTREAM_API_KEY 不是成对配置"
  echo "        UPSTREAM_BASE_URL='${base}'"
  if [[ -n "$key" ]]; then
    echo "        UPSTREAM_API_KEY='***set***'"
  else
    echo "        UPSTREAM_API_KEY=''"
  fi
  exit 1
fi

if [[ "$base" == *"openrouter.ai"* ]]; then
  echo "[INFO] 检测到 OpenRouter 配置: $base"
  if [[ "$base" != "https://openrouter.ai/api/v1" ]]; then
    echo "[WARN] OpenRouter 推荐使用 https://openrouter.ai/api/v1"
  fi
else
  echo "[INFO] 非 OpenRouter 配置: $base"
  echo "       请确认 key 与 base URL 属于同一平台。"
fi

url="${base%/}/chat/completions"
if [[ "$base" == */chat/completions ]]; then
  url="$base"
fi

echo "[INFO] smoke test -> $url"
http_code=$(curl -sS -o /tmp/upstream_smoke.json -w "%{http_code}" \
  -H "Authorization: Bearer $key" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: http://localhost" \
  -H "X-Title: gateway-smoke-test" \
  "$url" \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8}")

if [[ "$http_code" == "200" ]]; then
  echo "[PASS] /v1/chat/completions 最小请求通过 (HTTP 200)"
  exit 0
fi

echo "[FAIL] 最小请求失败，HTTP $http_code"
head -c 400 /tmp/upstream_smoke.json || true
echo
exit 2
