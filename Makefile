CUDNN_LIB := $(shell uv run python -c "import os, nvidia.cudnn; print(os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null)
CUBLAS_LIB := $(shell uv run python -c "import os, nvidia.cublas; print(os.path.join(nvidia.cublas.__path__[0], 'lib'))" 2>/dev/null)
CUDA_RT_LIB := $(shell uv run python -c "import os, nvidia.cuda_runtime; print(os.path.join(nvidia.cuda_runtime.__path__[0], 'lib'))" 2>/dev/null)
CURAND_LIB := $(shell uv run python -c "import os, nvidia.curand; print(os.path.join(nvidia.curand.__path__[0], 'lib'))" 2>/dev/null)
CUFFT_LIB := $(shell uv run python -c "import os, nvidia.cufft; print(os.path.join(nvidia.cufft.__path__[0], 'lib'))" 2>/dev/null)
export LD_LIBRARY_PATH := $(CUDNN_LIB):$(CUBLAS_LIB):$(CUDA_RT_LIB):$(CURAND_LIB):$(CUFFT_LIB):$(LD_LIBRARY_PATH)

.PHONY: test server run download

# 모델 다운로드 (Hugging Face)
download:
	uv run python scripts/model_download.py

# TTS 모델 테스트
test:
	uv run python scripts/test_tts_model.py

# FastAPI 서버 실행
server:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 임의 Python 스크립트 실행 (예: make run SCRIPT=scripts/foo.py)
run:
	uv run python $(SCRIPT)
