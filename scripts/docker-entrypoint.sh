#!/bin/bash
set -euo pipefail

_term() {
  kill -TERM "$API_PID" "$NGINX_PID" 2>/dev/null || true
  wait "$API_PID" "$NGINX_PID" 2>/dev/null || true
}
trap _term TERM INT

if [ -n "${CODENAMES_AI_CONFIG:-}" ]; then
  codenames-ai serve \
    --host 127.0.0.1 \
    --port 8001 \
    --no-static \
    --proxy-headers \
    --config "$CODENAMES_AI_CONFIG" &
else
  codenames-ai serve \
    --host 127.0.0.1 \
    --port 8001 \
    --no-static \
    --proxy-headers &
fi
API_PID=$!

nginx -c /app/deploy/nginx/default.conf -g 'daemon off;' &
NGINX_PID=$!

wait -n "$API_PID" "$NGINX_PID"
exit $?
