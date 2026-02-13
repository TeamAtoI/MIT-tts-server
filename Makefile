MODEL_REPO_ID ?= Supertone/supertonic-2
MODEL_REVISION ?= 75e6727618a02f323c720cba9478152d4bc16ca4
MODEL_LOCAL_DIR ?= ./assets

IMAGE_REPO ?= ghcr.io/teamatoi/mit-tts
TAG ?=
PLATFORM ?= linux/amd64

CUDNN_LIB := $(shell uv run --extra gpu python -c "import os, nvidia.cudnn; print(os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null)
CUBLAS_LIB := $(shell uv run --extra gpu python -c "import os, nvidia.cublas; print(os.path.join(nvidia.cublas.__path__[0], 'lib'))" 2>/dev/null)
CUDA_RT_LIB := $(shell uv run --extra gpu python -c "import os, nvidia.cuda_runtime; print(os.path.join(nvidia.cuda_runtime.__path__[0], 'lib'))" 2>/dev/null)
CURAND_LIB := $(shell uv run --extra gpu python -c "import os, nvidia.curand; print(os.path.join(nvidia.curand.__path__[0], 'lib'))" 2>/dev/null)
CUFFT_LIB := $(shell uv run --extra gpu python -c "import os, nvidia.cufft; print(os.path.join(nvidia.cufft.__path__[0], 'lib'))" 2>/dev/null)
export LD_LIBRARY_PATH := $(CUDNN_LIB):$(CUBLAS_LIB):$(CUDA_RT_LIB):$(CURAND_LIB):$(CUFFT_LIB):$(LD_LIBRARY_PATH)

.PHONY: download test server server-cpu run docker-login-ghcr docker-build-gpu docker-build-cpu docker-push-gpu docker-push-cpu docker-push-all

define require_tag
	@if [ -z "$(TAG)" ]; then \
		echo "TAG is required. Example: make $@ TAG=v0.1.0"; \
		exit 1; \
	fi
endef

# 모델 다운로드 (Hugging Face)
download:
	uv run python scripts/model_download.py \
		--repo-id "$(MODEL_REPO_ID)" \
		--revision "$(MODEL_REVISION)" \
		--local-dir "$(MODEL_LOCAL_DIR)"

# TTS 모델 테스트 (GPU)
test:
	uv run --extra gpu python scripts/test_tts_model.py

# FastAPI 서버 실행 (GPU 모드, 기본)
server:
	uv run --extra gpu uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# FastAPI 서버 실행 (CPU 모드)
server-cpu:
	TTS_USE_GPU=false uv run --extra cpu uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# GHCR 로그인 (환경 변수 필요: GHCR_USER, GHCR_TOKEN)
docker-login-ghcr:
	@if [ -z "$$GHCR_USER" ] || [ -z "$$GHCR_TOKEN" ]; then \
		echo "Set GHCR_USER and GHCR_TOKEN first."; \
		exit 1; \
	fi
	@echo "$$GHCR_TOKEN" | docker login ghcr.io -u "$$GHCR_USER" --password-stdin

docker-build-gpu:
	$(call require_tag)
	docker build --platform "$(PLATFORM)" \
		-f docker/Dockerfile.gpu \
		--build-arg MODEL_REPO_ID="$(MODEL_REPO_ID)" \
		--build-arg MODEL_REVISION="$(MODEL_REVISION)" \
		-t "$(IMAGE_REPO):gpu-$(TAG)" \
		.

docker-build-cpu:
	$(call require_tag)
	docker build --platform "$(PLATFORM)" \
		-f docker/Dockerfile.cpu \
		--build-arg MODEL_REPO_ID="$(MODEL_REPO_ID)" \
		--build-arg MODEL_REVISION="$(MODEL_REVISION)" \
		-t "$(IMAGE_REPO):cpu-$(TAG)" \
		.

docker-push-gpu:
	$(call require_tag)
	docker tag "$(IMAGE_REPO):gpu-$(TAG)" "$(IMAGE_REPO):gpu-latest"
	docker push "$(IMAGE_REPO):gpu-$(TAG)"
	docker push "$(IMAGE_REPO):gpu-latest"

docker-push-cpu:
	$(call require_tag)
	docker tag "$(IMAGE_REPO):cpu-$(TAG)" "$(IMAGE_REPO):cpu-latest"
	docker push "$(IMAGE_REPO):cpu-$(TAG)"
	docker push "$(IMAGE_REPO):cpu-latest"

docker-push-all: docker-push-gpu docker-push-cpu

# 임의 Python 스크립트 실행 (예: make run SCRIPT=scripts/foo.py)
run:
	uv run python $(SCRIPT)
