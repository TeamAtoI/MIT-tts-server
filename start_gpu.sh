#!/bin/bash
# Supertonic TTS Server - GPU 모드 시작 스크립트

# UV PATH 추가
export PATH="$HOME/.local/bin:$PATH"

# PyTorch cuDNN + Conda CUDA 라이브러리 경로
export LD_LIBRARY_PATH=/opt/conda/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# GPU 사용 활성화
export TTS_USE_GPU=true

# 서버 시작
cd "$(dirname "$0")"
echo "🚀 Starting Supertonic TTS Server with GPU acceleration..."
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs -I {} echo "   Device: {}" || echo "   Device: GPU not detected"
echo ""

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
