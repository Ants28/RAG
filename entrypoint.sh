#!/bin/bash
set -e  # Exit on any error

echo "Starting llama-server..."
llama-server -m ${MODEL_PATH} -c 2048 --host 0.0.0.0 --port 8080 > /var/log/llama-server.log 2>&1 &
LLAMA_PID=$!

echo "Waiting for llama-server to be ready..."
until curl -s -f http://localhost:8080/health > /dev/null; do
    echo "Still waiting for llama-server..."
    sleep 5
done

echo "llama-server is ready! Starting rag_server.py..."
exec python3 rag_server.py
