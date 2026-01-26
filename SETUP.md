# 환경 설정 가이드

이 문서는 Supertonic TTS 서버의 환경 설정 및 의존성 관리 방법을 상세히 설명합니다.

## 설치

### 1. uv 설치

**uv 설치**:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH 추가 (.bashrc에 영구 등록)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 버전 확인
uv --version
```

### 2. 모델 다운로드 (필수)

**모델 다운로드**:

Supertonic 2 ONNX 모델을 Hugging Face Hub을 통해 다운로드합니다:

```bash
cd supertonic-server
make download
```

다운로드되는 파일 구조:
```
assets/
├── onnx/
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── vector_estimator.onnx
│   ├── vocoder.onnx
│   ├── tts.json
│   └── unicode_indexer.json
└── voice_styles/
    ├── F1.json ~ F5.json
    └── M1.json ~ M5.json
```

> ⚠️ 모델 파일 없이는 서버가 실행되지 않습니다.  
> 📦 약 1.2GB의 데이터가 다운로드됩니다.  
> ✅ Git LFS 설치가 불필요하며, `huggingface-hub` 패키지를 사용하여 자동으로 다운로드됩니다.  
> ✅ 중단된 다운로드는 자동으로 재개됩니다.

### 3. 의존성 설치

```bash
cd supertonic-server

# GPU 버전 (기본, CUDA 11.8 + cuDNN 8.x)
uv sync

# CPU 전용 (GPU 없는 환경)
uv sync --extra cpu
```

> 📌 **CUDA 버전 요구사항**:
> - onnxruntime-gpu 1.16.3은 CUDA 11.8 + cuDNN 8.x 지원
> - GPU: NVIDIA GPU + CUDA 11.8 + cuDNN 8.x + 호환 드라이버 필요
> - CPU: GPU 없는 환경에서는 `--extra cpu` 옵션 사용

### 4. 사용법

```bash
# Makefile 타겟 사용 (권장)
make server      # GPU 모드로 서버 실행 (기본)
make server-cpu  # CPU 모드로 서버 실행
make test        # 모델 테스트

# 또는 환경 변수를 직접 설정
TTS_USE_GPU=false make server  # CPU 모드

# 또는 uv를 통한 직접 실행
uv run uvicorn app.main:app --port 8000
uv run python scripts/test_tts_model.py

# 또는 가상환경 활성화 후 사용
source .venv/bin/activate
uvicorn app.main:app --port 8000
python scripts/test_tts_model.py
```

> 💡 **GPU/CPU 모드 선택**:
> - 기본값은 GPU 모드입니다 (`make server`).
> - CPU 모드로 실행하려면 `make server-cpu`를 사용하세요.

### 5. 의존성 관리

```bash
# 패키지 추가
uv add <package-name>

# 패키지 제거
uv remove <package-name>

# 업데이트
uv sync --upgrade

# Lock 파일 재생성
uv lock --upgrade
```

---

## GPU 설정

> 💡 **기본 설정**: 이 프로젝트는 GPU 환경을 기본으로 합니다.  
> `uv sync`만 실행하면 자동으로 GPU 지원이 활성화됩니다.

### 요구사항

| 항목 | 요구사항 |
|------|----------|
| **GPU** | NVIDIA Tesla V100, A100, RTX 시리즈 등 |
| **CUDA** | **11.8** (onnxruntime-gpu 1.16.3) |
| **cuDNN** | **8.x** (CUDA 11.8 호환) |
| **드라이버** | 450.80.02+ (Linux), 452.39+ (Windows) |

### 1. CUDA 확인

```bash
# CUDA 버전 확인
nvidia-smi

# 예상 출력:
# CUDA Version: 12.2 (하위 호환으로 CUDA 11.8 사용 가능)
```

### 2. 의존성 설치 확인

```bash
# GPU 설정 (기본)
uv sync

# 설치 확인
uv run python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
# 출력 예: Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 3. cuDNN 라이브러리 확인

```bash
# PyTorch의 cuDNN 8.x (호환됨)
ls /opt/conda/lib/python3.10/site-packages/torch/lib/libcudnn*.so*
```

### 4. 환경변수 설정

`run_gpu.sh` 스크립트:

```bash
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"

# CUDA 라이브러리 경로 (시스템에 맞게 수정)
export LD_LIBRARY_PATH=/opt/conda/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# GPU 활성화
export TTS_USE_GPU=true

cd /data/ephemeral/home/supertonic-server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. GPU 작동 확인

```bash
# Python에서 CUDA 확인
uv run python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
"

# 서버 시스템 정보 확인
curl -s http://localhost:8000/tts/system | uv run python -m json.tool
```

**예상 출력**:
```json
{
  "execution_provider": "CUDAExecutionProvider",
  "available_providers": "CUDAExecutionProvider, CPUExecutionProvider",
  "device": "GPU",
  "sample_rate": 44100,
  "gpu_memory": "1008MB / 32768MB"
}
```

### 6. 서버 실행

```bash
# 백그라운드 실행
nohup ./run_gpu.sh > server_gpu.log 2>&1 &

# 로그 확인
tail -f server_gpu.log

# 프로세스 확인
ps aux | grep uvicorn
```

---

## 트러블슈팅

### libcublasLt.so.11: cannot open shared object file

**원인**: CUDA 라이브러리가 `LD_LIBRARY_PATH`에 없음

**해결**:
```bash
# 라이브러리 찾기
find /opt/conda -name "libcublasLt.so.11" 2>/dev/null
find /usr/local/cuda -name "libcublasLt.so.11" 2>/dev/null

# run_gpu.sh에 경로 추가
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
```
