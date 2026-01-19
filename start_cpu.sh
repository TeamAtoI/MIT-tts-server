#!/bin/bash
# Supertonic TTS Server - CPU 모드 시작 스크립트

# UV PATH 추가
export PATH="$HOME/.local/bin:$PATH"

# 서버 시작
cd "$(dirname "$0")"
echo "🚀 Starting Supertonic TTS Server (CPU mode)..."
echo ""

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
